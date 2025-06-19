import torch
import numpy as np  
from torch import nn, optim
from torch.nn import Module
import torch.nn.functional as F
from typing import List, Dict, Tuple, Callable
import logging
import random
import copy
import ray
from backend.server.server_config import AggregationStrategy, aggregation_methods, AttackType, attacks
from backend.dataset.dataset import FederatedDataLoader
from backend.client import Client , FedProxClient



@ray.remote
def train_client(client_id, client_loader, global_model, local_epochs, learning_rate, device, dp , fedprox , fedprox_mu, optimizer, momentum):
        '''
        # Initialize logger in the remote function
        logger = logging.getLogger(f"client_{client_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        
        logger.info(f"Training client {client_id}")
        '''
        if fedprox:
            client = FedProxClient(client_id, client_loader, dp , fedprox_mu)
        else:
            client = Client(client_id, client_loader, dp)
        return client.train(
            global_model=copy.deepcopy(global_model),
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            device=device,
        )    


class Server:
    """
    Base Server class for managing federated learning without any attacks.

    """

    def __init__(
        self,
        # Core FL components (required)
        federated_data_loader: FederatedDataLoader,  
        global_model: nn.Module,
        
        # FL Configuration (required)
        aggregation_strategy: AggregationStrategy,

        sampled: int,
        global_epochs: int,
        local_epochs: int,
        
        # Training hyperparameters
        learning_rate: float = 0.01,
        batch_size: int = 32,
        
        # Optimizer configuration
        optimizer: str = 'SGD',
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        
        # Advanced FL techniques
        local_dp: bool = False,
        fedprox: bool = False,
        fedprox_mu: float = 0.01,
        
        # System configuration
        device: str = 'cpu',
        
        # Attack configuration
        f: int = 0,
        **kwargs,
    ):
        # Core FL components
        self.federated_data_loader = federated_data_loader
        self.num_clients = self.federated_data_loader.dataset_handler.num_clients
        self.global_model = global_model
        self.global_parameters = self.global_model.state_dict()
        self.test_loader = self.federated_data_loader._get_test_data()
        
        # FL Configuration
        self.aggregation_strategy = aggregation_strategy

        self.sampled = sampled
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Optimizer configuration
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Advanced FL techniques
        self.local_dp = local_dp
        self.fedprox = fedprox
        self.fedprox_mu = fedprox_mu
        
        # System configuration
        self.device = device
        
        # Attack configuration
        self.f = f
        
        # Initialize aggregation method
        self.aggregation_method = aggregation_methods[aggregation_strategy]
        
        #additional methods kets etc
        self.trust_scores = {1 for _ in range(self.num_clients)}
        self.last_updates = {}
        self.trust_scores2 = {1 for _ in range(self.num_clients)}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.aggregation_kwargs = kwargs  
        

    def _order_clients_by_attackers(
        self, sampled_clients: List[int]
    ) -> Tuple[List[int], int]:
        """
        Re-orders the sampled clients by placing attackers (client_id < self.f)
        first and returns the ordered list along with the number of attackers.
        """
        attackers = [cid for cid in sampled_clients if cid < self.f]
        number_of_attackers_epoch = len(attackers)
        non_attackers = [cid for cid in sampled_clients if cid >= self.f]
        return attackers + non_attackers, number_of_attackers_epoch
    def _train_clients(self, sampled_client_ids: List[int]) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Launch parallel training tasks for the given client IDs and return their updates.

        Returns:
            List[Tuple[int, Dict[str, torch.Tensor]]]: A list of tuples pairing the client ID with its update.
        """
        if self.device == 'cuda':
            train_func = train_client.options(num_gpus=0.2)
        else:
            train_func = train_client

        futures = [
            train_func.remote(
                client_id,
                self.federated_data_loader._get_client_data(client_id),
                copy.deepcopy(self.global_model),
                self.local_epochs,
                self.learning_rate,
                self.device,
                self.local_dp,
                self.fedprox,
                self.fedprox_mu, 
                self.optimizer,
                self.momentum
            )
            for client_id in sampled_client_ids
        ]
        client_updates = ray.get(futures)
        return list(zip(sampled_client_ids, client_updates))


    def _sample_clients(self, num_of_sampled: int) -> Tuple[List[int], int]:
        """
        Sample a subset of clients uniformly.
        """
        if num_of_sampled > self.num_clients:
            self.logger.warning(
                "Requested number of clients exceeds available clients. Sampling all clients."
            )
        sampled_clients = random.sample(range(self.num_clients), num_of_sampled)
        ordered_clients, number_of_attackers_epoch = self._order_clients_by_attackers(sampled_clients)
        self.logger.info(f"Sampled clients: {ordered_clients}")
        return ordered_clients, number_of_attackers_epoch

    def _sample_from_weights(self, weights: Dict[int, float], num_of_sampled: int) -> Tuple[List[int], int]:
        """
        Sample a subset of clients based on their weights.
        """
        if num_of_sampled > self.num_clients:
            self.logger.warning(
                "Requested number of clients exceeds available clients. Sampling all clients."
            )
            
            
        # Filter out clients with weight 0
        filtered_weights = {client_id: weight for client_id, weight in weights.items() if weight > 0}
        
        available = len(filtered_weights)
        if num_of_sampled > available:
            self.logger.warning(
                f"Requested number of clients ({num_of_sampled}) exceeds available clients with non-zero weight ({available}). Sampling all available clients."
            )
            num_of_sampled = available
        total_weight = sum(filtered_weights.values())
        normalized_weights = {client_id: weight / total_weight for client_id, weight in filtered_weights.items()}
        
        sampled_clients = np.random.choice(
            a=list(normalized_weights.keys()),
            p=list(normalized_weights.values()),
            size=num_of_sampled,
            replace=False
        )
        ordered_clients, number_of_attackers_epoch = self._order_clients_by_attackers(list(sampled_clients))
        self.logger.info(f"Sampled clients: {ordered_clients}")
        return ordered_clients, number_of_attackers_epoch

    def run_federated_training(self):
        self.logger.info("Starting federated training.")
        for epoch in range(1, self.global_epochs + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            # Choose sampling method as needed (here we use sample_clients)
            sampled_client_ids, _ = self._sample_clients(self.sampled)
            # Launch parallel training tasks using the helper method
            client_updates = self._train_clients(sampled_client_ids)
            self._aggregate_client_updates(client_updates)
            accuracy = self._evaluate_global_model(epoch)
            self.logger.info(f"\033[94mGlobal Epoch {epoch} completed.\033[0m")
        self.logger.info("Federated training completed.")
        return accuracy

    def _aggregate_client_updates(self, gradients: List[Tuple[int, Dict[str, torch.Tensor]]]):
        """
        Aggregate client updates to update the global model.

        Args:
            gradients (List[tuple]): List of tuples containing client IDs and their update dictionaries.
        """
        self.logger.info("Aggregating client updates.")

        aggregation_kwargs = self.aggregation_kwargs.copy() # Copy to avoid modifying the original dict
        
        if self.aggregation_strategy in {AggregationStrategy.KeTS, AggregationStrategy.KeTSV2 ,  AggregationStrategy.KeTS_MedTrim, AggregationStrategy.Testing}:
            aggregation_kwargs.update({
            'trust_scores': self.trust_scores,
            'last_updates': self.last_updates,
            'baseline_decreased_score': 0.02,
            'last_global_update': self,
            'trust_scores2': self.trust_scores2,
            'clusters': self.federated_data_loader.dataset_handler.clusters, #TESTING
            })

        
        try:
            self.aggregation_method(
                gradients=gradients,
                net=self.global_model,
                lr=self.learning_rate,
                f=self.f,
                device=torch.device(self.device),
                **aggregation_kwargs  # data_sizes included conditionally
            )
            self.logger.info("Client updates aggregated successfully.")
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise

    def _evaluate_global_model(self, epoch: int):
        """
        Evaluate the global model on the test dataset.

        Args:
            epoch (int): The current global epoch number.
        """
        self.logger.info(f"Evaluating global model at Global Epoch {epoch}.")
        self.global_model.eval()
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.test_loader:
                # Handle batch as dict (image data) or tuple/list (tabular data)
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.squeeze().to(self.device)
                else:
                    raise ValueError("Unexpected batch format in test_loader. Expected dict with keys ('image','label') or tuple (features, labels).")
                
                outputs = self.global_model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        self.logger.info(f"Epoch {epoch} - Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return accuracy

    def _save_global_model(self, filepath: str):
        """
        Save the global model to a file.

        Args:
            filepath (str): The path to save the global model.
        """
        torch.save(self.global_model.state_dict(), filepath)
        self.logger.info(f"Global model saved to {filepath}.")

    def _load_global_model(self, filepath: str):
        """
        Load the global model from a file.

        Args:
            filepath (str): The path to load the global model from.
        """
        self.global_model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.global_model.to(self.device)
        self.global_parameters = self.global_model.state_dict()
        self.logger.info(f"Global model loaded from {filepath}.")


class AttackServer(Server):
    def __init__(self, attack_type: AttackType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_method = attacks[attack_type]
        if self.aggregation_strategy in {AggregationStrategy.KeTS, AggregationStrategy.KeTSV2}:
            self.trust_scores = {cid: 1.0 for cid in range(self.num_clients)}
            self.last_updates = {cid: None for cid in range(self.num_clients)}
            self.last_global_update = None
            
            self.trust_scores2 = {cid: 1.0 for cid in range(self.num_clients)}
    
    def _compute_attack(self, updates, lr,num_attackers_epoch ,f):
        self.logger.info(f"{self.attack_method.__name__} attack computed.")
        if self.attack_method.__name__ in ['krum_attack', 'trim_attack']:
            return self.attack_method(updates, self.global_model, lr, f,num_attackers_epoch ,self.device)
        else:
            return self.attack_method(updates, lr, f, num_attackers_epoch ,self.device)

    def run_federated_training(self):
        self.logger.info("Starting federated training with attacks.")
        self.epoch_results = []  
        for epoch in range(1, self.global_epochs + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            if self.aggregation_strategy in {AggregationStrategy.KeTS, AggregationStrategy.KeTSV2}:
                sampled_client_ids , number_of_attackers_epoch = self._sample_from_weights(self.trust_scores, self.sampled)
            else:   
                sampled_client_ids , number_of_attackers_epoch = self._sample_clients(self.sampled)
            # Use the common training helper
            client_updates = self._train_clients(sampled_client_ids)
            
            # Extract updates, compute attacked updates, and update client_updates accordingly
            updates_list = [update['flattened_diffs'] for (_, update) in client_updates]
            attacked_updates = self._compute_attack(updates_list, self.learning_rate,  number_of_attackers_epoch ,self.f)
            for i, (cid, update) in enumerate(client_updates):
                client_updates[i] = (cid, {'flattened_diffs' : attacked_updates[i] , 'data_size' : update['data_size']})

            self._aggregate_client_updates(client_updates)
            if self.aggregation_strategy in {AggregationStrategy.KeTS, AggregationStrategy.KeTSV2}:
                for cid, attacked_update in client_updates:
                    if self.last_updates[cid] is None:
                        self.last_updates[cid] = attacked_update['flattened_diffs']
                    else:
                        ema_alpha = 1
                        self.last_updates[cid] = (
                            ema_alpha * attacked_update['flattened_diffs'] +
                            (1 - ema_alpha) * self.last_updates[cid]
                        )
                
            
    
            accuracy = self._evaluate_global_model(epoch)
            self.epoch_results.append((epoch, accuracy)) 

            self.logger.info(f"Global Epoch {epoch} completed.")
        self.logger.info("Federated training with attacks completed.")
        return accuracy

class FLTrustServer(AttackServer):
    """
    FLTrust server variant that trains on a local trusted dataset using the same
    hyperparameters as clients. The server update is used as a reference to re-weight
    client updates based on cosine similarity.
    """
    def __init__(self, server_data_loader, *args, **kwargs):
        """
        Initialize FLTrust server.

        Args:
            server_data_loader: DataLoader for the server's trusted local dataset.
            *args, **kwargs: Additional arguments passed to AttackServer.
        """
        super().__init__(*args, **kwargs)
        self.server_data_loader = server_data_loader
        self.aggregation_method = None
        
    
    def _fl_train_server(
        self,
        global_model: Module,
        local_epochs: int,
        learning_rate: float,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Train the global model on the client's local data.

        Args:
            global_model (Module): The global model to be trained.
            local_epochs (int): Number of local epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on ('cpu' or 'cuda').

        Returns:
            Dict[str, torch.Tensor]: The updated model parameters after training.
        """
        self.logger.info(f"Server started training.")
        

        
        server_model = copy.deepcopy(global_model).to(device)
        server_model.load_state_dict(global_model.state_dict())
        
        
        server_model.train()
        # Define optimizer and loss function
        optimizer = optim.SGD(server_model.parameters(), lr=learning_rate , momentum= 0.9)
        #optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Conditional AMP setup
        use_amp = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        

        # Clone initial parameters for computing updates
        initial_params = {key: param.clone() for key, param in server_model.named_parameters()}
        
        for epoch in range(local_epochs):
            
            for batch_idx, batch in enumerate(self.server_data_loader):
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)
                    targets = batch['label'].to(device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    if targets.ndim == 2 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    targets = targets.to(device)
                else:
                    raise ValueError("Unexpected batch format. Expected dict with keys ('image','label') or tuple (features, labels).")

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = server_model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = server_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
        
        final_params = {key: param.clone() for key, param in server_model.named_parameters()}
        param_differences = {
            key: (final_params[key] - initial_params[key]) for key in final_params.keys()
        }

        gradient_like_diffs = torch.cat([diff.view(-1) for diff in param_differences.values()]).unsqueeze(1)

        
        return {
            'flattened_diffs': gradient_like_diffs,
            'data_size': len(self.server_data_loader.dataset)        
        }
    
    def _fltrust_aggregation(self, gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
                              net: torch.nn.Module,
                              lr: float,
                              f: int,
                              device: torch.device,
                              **kwargs) -> None:
        """
        FLTrust aggregation method based on cosine similarity with a trusted server update.
        
        Args:
            gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): Client updates;
                the last element is assumed to be the trusted server update.
            net (torch.nn.Module): The global model.
            lr (float): Learning rate.
            f (int): Number of malicious clients (first f clients).
            device (torch.device): Computation device.
            **kwargs: Additional parameters if needed.
        """
        # Extract the flattened updates from each client update dictionary.
        param_list = [upd["flattened_diffs"] for _, upd in gradients]
        # Assume the last update is the trusted server update.
        n = len(param_list) - 1  
        baseline = param_list[-1].squeeze()
        sim_list = []
        new_param_list = []

        # Compute cosine similarities between baseline and each client update.
        for grad in param_list:
            grad_squeezed = grad.squeeze()
            cosine_sim = torch.dot(baseline, grad_squeezed) / (
                (torch.norm(baseline) + 1e-9) * (torch.norm(grad_squeezed) + 1e-9)
            )
            sim_list.append(cosine_sim)
        # Remove the similarity for the server update.
        sim_tensor = torch.stack(sim_list)[:-1]

        # Clip using ReLU and normalize to obtain trust scores.
        sim_tensor = F.relu(sim_tensor)
        normalized_weights = sim_tensor / (torch.sum(sim_tensor).item() + 1e-9)

        # For each client update, re-scale it by its trust score and the magnitude of the baseline.
        for i in range(n):
            new_param = (
                param_list[i]
                * normalized_weights[i]
                / (torch.norm(param_list[i]) + 1e-9)
                * torch.norm(baseline)
            )
            new_param_list.append(new_param)

        # Compute the aggregated (global) update as the sum over weighted client updates.
        global_update = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)
        # Update the global model's parameters.
        with torch.no_grad():
            idx = 0
            for param in net.parameters():
                numel = param.numel()
                param.add_(global_update[idx: idx + numel].view(param.size()), alpha=1)
                idx += numel
        

    def run_federated_training(self):
        self.epoch_results = []  # new: initialize results list
        self.logger.info("Starting federated training with attacks.")
        for epoch in range(1, self.global_epochs + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            # FLTrust may use a different sampling method; here we use sample_clients
            sampled_client_ids, number_of_attackers_epoch = self._sample_clients(self.sampled)
            
            # Use the common training helper
            client_updates = self._train_clients(sampled_client_ids)
            
            # Generate the server's trusted update and add it to client updates
            server_update = self._fl_train_server(
                global_model=copy.deepcopy(self.global_model),
                local_epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                device=torch.device(self.device),
            )
            client_updates.append((-1, server_update))
            
            self._fltrust_aggregation(
                client_updates,
                self.global_model,
                self.learning_rate,
                self.f,
                torch.device(self.device)
            )
            accuracy = self._evaluate_global_model(epoch)
            self.epoch_results.append((epoch, accuracy))  
            
        self.logger.info("Federated training with attacks completed.")
        return accuracy