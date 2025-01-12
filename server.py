import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
from aggregation_techniques.aggregation import (
    fedavg,
    krum,
    median_aggregation,
    trim_mean,
)
from enum import Enum, auto
import copy
import random


from dataset import FederatedDataLoader
from client import Client


class AggregationStrategy(Enum):
    FEDAVG = auto()
    KRUM = auto()
    MEDIAN = auto()
    TRIM_MEAN = auto()
#Enumeration that is later mapped to each aggregation method. The auto() function is used to automatically assign unique values to each member.

aggregation_methods = {
    AggregationStrategy.FEDAVG: fedavg,
    AggregationStrategy.KRUM: krum,
    AggregationStrategy.MEDIAN: median_aggregation,
    AggregationStrategy.TRIM_MEAN: trim_mean
}


class Server:
    """
    Base Server class for managing federated learning without any attacks.

    Attributes:
        client_ids (List[int]): List of client IDs.
        global_epoch (int): Current global epoch number.
        local_epochs (int): Number of local epochs for each client.
        learning_rate (float): Learning rate for client optimizers.
        batch_size (int): Batch size for client data loaders.
        device (str): Device to run the training on ('cpu' or 'cuda').
        federated_data_loader (FederatedDataLoader): DataLoader for federated datasets.
        test_loader (DataLoader): DataLoader for the server's test dataset.
        global_model (nn.Module): The global model to be trained.
        global_parameters (Dict[str, torch.Tensor]): State dictionary of the global model.
        logger (logging.Logger): Logger for the server.
        aggregation_method (Callable): Aggregation method to use.
        aggregation_kwargs (Dict): Additional keyword arguments for the aggregation method.
    """

    def __init__(
        self,
        federated_data_loader: FederatedDataLoader,  
        global_model: nn.Module,
        aggregation_strategy: AggregationStrategy,
        global_epochs: int = 0,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize the Server.

        Args:
            client_ids (List[int]): List of client IDs.
            federated_data_loader (FederatedDataLoader): Instance of FederatedDataLoader.
            global_model (nn.Module): The global model to be trained.
            aggregation_strategy (AggregationStrategy): Selected aggregation strategy.
            global_epochs (int, optional): Starting global epoch. Defaults to 0.
            local_epochs (int, optional): Number of local epochs for training. Defaults to 1.
            learning_rate (float, optional): Learning rate for optimizers. Defaults to 0.01.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
            **kwargs: Additional arguments for the aggregation method.
        """
        
        self.global_epoch = global_epochs
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

        self.federated_data_loader = federated_data_loader
        self.num_clients = self.federated_data_loader.dataset_handler.num_clients
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
            num_clients (int): Number of clients to sample.

        Returns:
            List[int]: List of sampled client IDs.
        """
        if num_of_sampled > self.num_clients:
            self.logger.warning("Requested number of clients exceeds available clients. Sampling all clients.")
        sampled_clients = random.sample(range(self.num_clients), num_of_sampled)
        self.logger.info(f"Sampled clients: {sampled_clients}")
        return sampled_clients

    def run_federated_training(self, total_epochs: int, sampled_clients: int):
        """
        Run the federated training process.

        Args:
            total_epochs (int): Total number of global epochs to run.
        """
        self.logger.info("Starting federated training.")
        for epoch in range(1, total_epochs + 1):
            self.logger.info(f"Global Epoch {epoch} started.")
            sampled_client_ids = self.sample_clients(sampled_clients)  # Modify as needed

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
            self.evaluate_global_model(epoch)
            self.logger.info(f"\033[94mGlobal Epoch {epoch} completed.\033[0m")
        self.logger.info("Federated training completed.")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, Dict[str, torch.Tensor]]]):
        """
        Aggregate client updates to update the global model.

        Args:
            client_updates (List[tuple]): List of tuples containing client IDs and their update dictionaries.
        """
        self.logger.info("Aggregating client updates.")

        # Extract gradients and data_sizes from client_updates
        gradients = [update['flattened_diffs'] for _, update in client_updates]
        data_sizes = [update.get('data_size', 1) for _, update in client_updates]  # Default to 1 if not provided

        # Prepare keyword arguments
        aggregation_kwargs = self.aggregation_kwargs.copy()
        
        # Include data_sizes only for FedAvg
        if self.aggregation_method.__name__ == 'fedavg':
            aggregation_kwargs['data_sizes'] = data_sizes

        # Call the selected aggregation method
        try:
            self.aggregation_method(
                gradients=gradients,
                net=self.global_model,
                lr=self.learning_rate,
                f=self.aggregation_kwargs.get('f', 0),
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
    """
    Server subclass that introduces attacks during aggregation.
    
    Inherits from Server and overrides the aggregation method to include attack logic.
    """

    def __init__(
        self,
        client_ids: List[int],
        federated_data_loader: FederatedDataLoader,
        global_model: nn.Module,
        global_epochs: int = 0,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = 'cpu',
        attack_type: Optional[str] = None
    ):
        """
        Initialize the AttackServer.
        
        Args:
            client_ids (List[int]): List of client IDs.
            federated_data_loader (FederatedDataLoader): Instance of FederatedDataLoader.
            global_model (Module): The global model to be trained.
            global_epochs (int, optional): Starting global epoch. Defaults to 0.
            local_epochs (int, optional): Number of local epochs for training. Defaults to 1.
            learning_rate (float, optional): Learning rate for optimizers. Defaults to 0.01.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
            attack_type (str, optional): Type of attack to perform. Defaults to None.
        """
        super().__init__(
            client_ids=client_ids,
            federated_data_loader=federated_data_loader,
            global_model=global_model,
            aggregation_strategy=AggregationStrategy.FEDAVG,  # Default to FEDAVG for AttackServer
            global_epochs=global_epochs,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device
        )
        self.attack_type = attack_type
        if self.attack_type:
            self.logger.info(f"AttackServer initialized with attack type: {self.attack_type}")

    def aggregate_client_updates(self, client_updates: List[tuple]):
        """
        Aggregate client updates with attack considerations.
        
        Args:
            client_updates (List[tuple]): List of tuples containing client IDs and their model state dictionaries.
        """
        self.logger.info("Aggregating client updates with attack considerations.")
        
        # Implement attack logic based on attack_type
        if self.attack_type == "malicious":
            self.logger.warning("Malicious attack detected. Modifying client updates.")
            for i, (client_id, update) in enumerate(client_updates):
                # Example modification: Add noise to client updates
                for key in update:
                    noise = torch.randn_like(update[key]) * 0.01
                    client_updates[i] = (client_id, update[key] + noise)
            self.logger.info("Client updates have been tampered with for malicious attack.")
        
        # Call the base aggregation method
        super().aggregate_client_updates(client_updates)
        self.logger.info("Aggregation with attacks completed.")


class TrustServer(Server):
    """
    Server subclass that uses trust scores for aggregating client updates.
    
    Inherits from Server and overrides the aggregation method to incorporate trust scores.
    """

    def __init__(
        self,
        client_ids: List[int],
        federated_data_loader: FederatedDataLoader,
        global_model: nn.Module,
        global_epochs: int = 0,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize the TrustServer.
        
        Args:
            client_ids (List[int]): List of client IDs.
            federated_data_loader (FederatedDataLoader): Instance of FederatedDataLoader.
            global_model (Module): The global model to be trained.
            global_epochs (int, optional): Starting global epoch. Defaults to 0.
            local_epochs (int, optional): Number of local epochs for training. Defaults to 1.
            learning_rate (float, optional): Learning rate for optimizers. Defaults to 0.01.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
        """
        super().__init__(
            client_ids=client_ids,
            federated_data_loader=federated_data_loader,
            global_model=global_model,
            aggregation_strategy=AggregationStrategy.FEDAVG,  # Default to FEDAVG for TrustServer
            global_epochs=global_epochs,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device
        )
        self.trust_scores: Dict[int, float] = {client_id: 1.0 for client_id in self.client_ids}
        self.logger.info("TrustServer initialized with trust scores.")

    def aggregate_client_updates(self, client_updates: List[tuple]):
        """
        Aggregate client updates using trust scores.
        
        Args:
            client_updates (List[tuple]): List of tuples containing client IDs and their model state dictionaries.
        """
        self.logger.info("Aggregating client updates using trust scores.")
        
        # Implement aggregation logic based on trust scores
        self.logger.info("Global model updated with trust-weighted client parameters.")

    def update_trust_score(self, client_id: int):
        """
        Update the trust score for a specific client based on performance.
        
        Args:
            client_id (int): The ID of the client.
            performance_metric (float): The performance metric to adjust trust score.
        """
        # Example: Simple update rule (can be replaced with a more sophisticated method)
        
        self.logger.info(f"Updated trust score for client {client_id}.")