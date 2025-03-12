import torch
import numpy as np  
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
from torch.nn import Module
import torch.nn.functional as F
from torch import optim
from aggregation_techniques.aggregation import (
    fedavg,
    krum,
    median_aggregation,
    trim_mean,
    KeTS,
    DCT,
    DCT_K,
    DCT_raw,
    
)
from enum import Enum, auto
import copy
import random
import ray  

from dataset import FederatedDataLoader
from client import Client
from attacks import min_max_attack, min_sum_attack, krum_attack, trim_attack, no_attack , gaussian_attack , label_flip_attack , min_max_attack_variant , sign_flip_attack


class AggregationStrategy(Enum):
    FEDAVG = auto()
    KRUM = auto()
    MEDIAN = auto()
    TRIM_MEAN = auto()
    KeTS = auto()
    DCT = auto()
    FLTRUST = auto()
    DCT_K = auto()
    DCT_raw = auto()
#Enumeration that is later mapped to each aggregation method. The auto() function is used to automatically assign unique values to each member.

aggregation_methods = {
    AggregationStrategy.FEDAVG: fedavg,
    AggregationStrategy.KRUM: krum,
    AggregationStrategy.MEDIAN: median_aggregation,
    AggregationStrategy.TRIM_MEAN: trim_mean,
    AggregationStrategy.KeTS: KeTS,
    AggregationStrategy.DCT: DCT,
    AggregationStrategy.FLTRUST: None,
    AggregationStrategy.DCT_K: DCT_K,
    AggregationStrategy.DCT_raw: DCT_raw,
    
}

class AttackType(Enum):
    NO_ATTACK = auto()
    MIN_MAX = auto()
    MIN_SUM = auto()
    KRUM = auto()
    TRIM = auto()
    GAUSSIAN = auto()
    LABEL_FLIP = auto()
    MIN_MAX_V2 = auto()
    SIGN_FLIP = auto()
    
#Enumeration that is later mapped to each attack method. The auto() function is used to automatically assign unique values to each member.

attacks = {
    AttackType.MIN_MAX: min_max_attack,
    AttackType.MIN_SUM: min_sum_attack,
    AttackType.KRUM: krum_attack,
    AttackType.TRIM: trim_attack,
    AttackType.NO_ATTACK: no_attack,
    AttackType.GAUSSIAN: gaussian_attack,
    AttackType.LABEL_FLIP: label_flip_attack,
    AttackType.MIN_MAX_V2: min_max_attack_variant,
    AttackType.SIGN_FLIP: sign_flip_attack
    
}
@ray.remote
def train_client(client_id, client_loader, global_model, local_epochs, learning_rate, device , dp):
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
        client = Client(client_id, client_loader,dp)
        return client.train(
            global_model=copy.deepcopy(global_model),
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )    


class Server:
    """
    Base Server class for managing federated learning without any attacks.

    """

    def __init__(
        self,
        federated_data_loader: FederatedDataLoader,  
        global_model: nn.Module,
        aggregation_strategy: AggregationStrategy,
        sampled: int,
        global_epochs: int = 0,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        local_dp: bool = False,
        device: str = 'cpu',
        f: int = 0,
        **kwargs
    ):
        """
        Initialize the Server.

        Args:
            federated_data_loader (FederatedDataLoader): Instance of FederatedDataLoader.
            global_model (nn.Module): The global model to be trained.
            aggregation_strategy (AggregationStrategy): Selected aggregation strategy.
            global_epochs (int, optional): Starting global epoch. Defaults to 0.
            local_epochs (int, optional): Number of local epochs for training. Defaults to 1.
            learning_rate (float, optional): Learning rate for optimizers. Defaults to 0.01.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
            f (int, optional): Number of malicious clients. Defaults to 0.
            **kwargs: Additional arguments for the aggregation method.
        """
        
        self.global_epoch = global_epochs
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.f = f
        self.local_dp = local_dp

        self.federated_data_loader = federated_data_loader
        self.num_clients = self.federated_data_loader.dataset_handler.num_clients
        self.sampled = sampled # Number of clients to sample per round
        self.test_loader = self.federated_data_loader.get_test_data()

        self.global_model = global_model.to(self.device)
        self.global_parameters = self.global_model.state_dict()

        self.logger = logging.getLogger(__name__)
        self.logger.info("Server initialized.")

        # Set the aggregation method based on strategy
        if aggregation_strategy not in aggregation_methods:
            self.logger.error(f"Aggregation strategy {aggregation_strategy} is not supported.")
            raise ValueError(f"Unsupported aggregation strategy: {aggregation_strategy}")

        self.aggregation_method = aggregation_methods[aggregation_strategy]
        self.aggregation_kwargs = kwargs  # Store additional kwargs for aggregation

    def sample_clients(self, num_of_sampled : int) -> List[int]:
        """
        Sample a subset of clients for the current round.

        Args:
            num_of_sampled (int): Number of clients to sample.

        Returns:
            List[int]: List of sampled client IDs.
        """
        if num_of_sampled > self.num_clients:
            self.logger.warning("Requested number of clients exceeds available clients. Sampling all clients.")
        sampled_clients = random.sample(range(self.num_clients), num_of_sampled)
        attackers = [cid for cid in sampled_clients if cid < self.f]
        number_of_attackers_epoch = len(attackers)
        non_attackers = [cid for cid in sampled_clients if cid >= self.f]
        sampled_clients = attackers + non_attackers #attackers are always sampled first
        self.logger.info(f"Sampled clients: {sampled_clients}")
        return sampled_clients , number_of_attackers_epoch
    def sample_from_weights(self, weights: Dict[int,float], num_of_sampled: int) -> List[int]:
        """
        Sample a subset of clients based on their weights.

        Args:
            weights (Dict[int, float]): Dictionary of client IDs and their weights.
            num_of_sampled (int): Number of clients to sample.

        Returns:
            List[int]: List of sampled client IDs.
        """
        if num_of_sampled > self.num_clients:
            self.logger.warning("Requested number of clients exceeds available clients. Sampling all clients.")
        # Filter out clients with weight 0
        filtered_weights = {client_id: weight for client_id, weight in weights.items() if weight > 0}
        
        # Normalize weights
        total_weight = sum(filtered_weights.values())
        normalized_weights = {client_id: weight / total_weight for client_id, weight in filtered_weights.items()}
        
        # Sample clients based on normalized weights
        sampled_clients = np.random.choice(
            a=list(normalized_weights.keys()),
            p=list(normalized_weights.values()),
            size=num_of_sampled,
            replace=False
        )
        attackers = [cid for cid in sampled_clients if cid < self.f]
        number_of_attackers_epoch = len(attackers)
        non_attackers = [cid for cid in sampled_clients if cid >= self.f]
        sampled_clients = attackers + non_attackers #attackers are always sampled first
        self.logger.info(f"Sampled clients: {sampled_clients}")
        return sampled_clients , number_of_attackers_epoch
        

    def run_federated_training(self):
        """
        Run the federated training process.

        """
        self.logger.info("Starting federated training.")
        for epoch in range(1, self.global_epoch + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            sampled_client_ids , number_of_attackers_epoch = self.sample_clients(self.sampled)  # Modify as needed

            client_updates: List[Tuple[int, Dict[str, torch.Tensor]]] = []
            for client_id in sampled_client_ids:
                self.logger.info(f"Training client {client_id} for {self.local_epochs} local epochs.")
                # Instantiate the Client with client_loader
                client_loader = self.federated_data_loader.get_client_data(
                    client_id=client_id,
                    
                )
                client = Client(client_id, client_loader)
                client_update = client.train(
                    global_model=copy.deepcopy(self.global_model),  # Pass a copy of the global model
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    device=torch.device(self.device),
                )
                client_updates.append((client_id, client_update))
                # Client object is discarded after training

            self.aggregate_client_updates(client_updates)
            accuracy = self.evaluate_global_model(epoch)
            self.logger.info(f"\033[94mGlobal Epoch {epoch} completed.\033[0m")
        self.logger.info("Federated training completed.")
        return accuracy
        

    def aggregate_client_updates(self, gradients: List[Tuple[int, Dict[str, torch.Tensor]]]):
        """
        Aggregate client updates to update the global model.

        Args:
            gradients (List[tuple]): List of tuples containing client IDs and their update dictionaries.
        """
        self.logger.info("Aggregating client updates.")

        aggregation_kwargs = self.aggregation_kwargs.copy() # Copy to avoid modifying the original dict
        
        if self.aggregation_method.__name__ == 'KeTS' or self.aggregation_method.__name__ == 'DCT' or self.aggregation_method.__name__ == 'DCT_raw':
            aggregation_kwargs['trust_scores'] = self.trust_scores
            aggregation_kwargs['last_updates'] = self.last_updates
            aggregation_kwargs['baseline_decreased_score'] = 0.005
            aggregation_kwargs['last_global_update'] = self

        # Call the selected aggregation method
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

    def evaluate_global_model(self, epoch: int):
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
                inputs, labels = batch['image'].to(self.device), batch['label'].to(self.device)
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

    def save_global_model(self, filepath: str):
        """
        Save the global model to a file.

        Args:
            filepath (str): The path to save the global model.
        """
        torch.save(self.global_model.state_dict(), filepath)
        self.logger.info(f"Global model saved to {filepath}.")

    def load_global_model(self, filepath: str):
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
        if self.aggregation_method == KeTS or self.aggregation_method == DCT or self.aggregation_method == DCT_raw:
            self.trust_scores = {cid : 1.0 for cid in range(self.num_clients)}
            self.last_updates = {cid : None for cid in range(self.num_clients)}
            self.last_global_update = None
    
    def compute_attack(self, updates, lr,num_attackers_epoch ,f):
        self.logger.info(f"{self.attack_method.__name__} attack computed.")
        if self.attack_method.__name__ in ['krum_attack', 'trim_attack']:
            return self.attack_method(updates, self.global_model, lr, f,num_attackers_epoch ,self.device)
        else:
            return self.attack_method(updates, lr, f, num_attackers_epoch ,self.device)

    def run_federated_training(self):
        self.logger.info("Starting federated training with attacks.")
        for epoch in range(1, self.global_epoch + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            if self.aggregation_method == KeTS:
                sampled_client_ids , number_of_attackers_epoch = self.sample_from_weights(self.trust_scores, self.sampled)
            else:   
                sampled_client_ids , number_of_attackers_epoch = self.sample_clients(self.sampled)
            '''
            client_updates = []
            for client_id in sampled_client_ids:
                client_loader = self.federated_data_loader.get_client_data(client_id=client_id)
                client = Client(client_id, client_loader)
                client_update = client.train(
                    global_model=copy.deepcopy(self.global_model),
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    device=torch.device(self.device),
                )
                client_updates.append((client_id, client_update))
            '''
            # Launch parallel training tasks
            futures = [
                train_client.remote(
                    client_id,
                    self.federated_data_loader.get_client_data(client_id),
                    copy.deepcopy(self.global_model),
                    self.local_epochs,
                    self.learning_rate,
                    self.device,
                    self.local_dp
                )
                for client_id in sampled_client_ids
            ]
            
            # Retrieve results
            client_updates = ray.get(futures)
            client_updates = list(zip(sampled_client_ids, client_updates))
            
            
            # Extract only the weight updates, compute attacked updates, then reassign
            updates_list = [update['flattened_diffs'] for (_, update) in client_updates]
            attacked_updates = self.compute_attack(updates_list, self.learning_rate,  number_of_attackers_epoch ,self.f)
            for i, (cid, update) in enumerate(client_updates):
                client_updates[i] = (cid, {'flattened_diffs' : attacked_updates[i] , 'data_size' : update['data_size']})

            self.aggregate_client_updates(client_updates)
            if self.aggregation_method == KeTS or self.aggregation_method == DCT or self.aggregation_method == DCT_raw:
                for cid, attacked_update in client_updates:
                    self.last_updates[cid] = attacked_update['flattened_diffs']
                
            
    
            accuracy = self.evaluate_global_model(epoch)

            self.logger.info(f"Global Epoch {epoch} completed.")
        self.logger.info("Federated training with attacks completed.")
        return accuracy

class FLTrust(AttackServer):
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
        
    
    def fl_train_server(
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
        

        # Create a new model and load global state
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
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = server_model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = server_model(images)
                    loss = criterion(outputs, labels)
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
    
    def fltrust_aggregation(self, gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
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
        self.logger.info("Starting federated training with attacks.")
        for epoch in range(1, self.global_epoch + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            if self.aggregation_method == KeTS:
                sampled_client_ids , number_of_attackers_epoch = self.sample_from_weights(self.trust_scores, self.sampled)
            else:   
                sampled_client_ids , number_of_attackers_epoch = self.sample_clients(self.sampled)
            '''
            client_updates = []
            for client_id in sampled_client_ids:
                client_loader = self.federated_data_loader.get_client_data(client_id=client_id)
                client = Client(client_id, client_loader)
                client_update = client.train(
                    global_model=copy.deepcopy(self.global_model),
                    local_epochs=self.local_epochs,
                    learning_rate=self.learning_rate,
                    device=torch.device(self.device),
                )
                client_updates.append((client_id, client_update))
            '''
            # Launch parallel training tasks
            futures = [
                train_client.remote(
                    client_id,
                    self.federated_data_loader.get_client_data(client_id),
                    copy.deepcopy(self.global_model),
                    self.local_epochs,
                    self.learning_rate,
                    self.device,
                    self.local_dp
                )
                for client_id in sampled_client_ids
            ]
            
            # Retrieve results
            client_updates = ray.get(futures)
            client_updates = list(zip(sampled_client_ids, client_updates))
            
            
            # Extract only the weight updates, compute attacked updates, then reassign
            updates_list = [update['flattened_diffs'] for (_, update) in client_updates]
            attacked_updates = self.compute_attack(updates_list, self.learning_rate,  number_of_attackers_epoch ,self.f)
            for i, (cid, update) in enumerate(client_updates):
                client_updates[i] = (cid, {'flattened_diffs' : attacked_updates[i] , 'data_size' : update['data_size']})

            server_update = self.fl_train_server(
                global_model=copy.deepcopy(self.global_model),
                local_epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                device=torch.device(self.device),
            )
            client_updates.append((-1,server_update))
            
            self.fltrust_aggregation(client_updates, self.global_model, self.learning_rate, self.f, torch.device(self.device)) 
            
    
            accuracy = self.evaluate_global_model(epoch)

            self.logger.info(f"Global Epoch {epoch} completed.")
        
        self.logger.info("Federated training with attacks completed.")
        return accuracy