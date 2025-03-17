import logging
from utils.logger import setup_logger

import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import Counter
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from typing import List

class DatasetUtils:
    @staticmethod
    def apply_transforms(batch) -> dict:
        """
        Apply transformations to the images in the dataset.
        """
        pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) #Mnist Value -> add fashion_mnist (0.1307,), (0.3081,
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    


class DatasetHandler:
    def __init__(self, datasetID: str, num_clients: int, partition_type: str , alpha: float , seed: int = 33):
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fds = None
        self.client_datasets = None
        self.datasetID = datasetID
        self.label_flipping_attack = False
        self.num_attackers = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetHandler initialized.")

    def _initialize_partitioner(self):
        """
        Initialize the partitioner based on the partition type.
        """
        if self.partition_type == 'iid':
            partitioner = IidPartitioner(num_partitions=self.num_clients)
        else:
            partitioner = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label", 
                                              alpha=self.alpha, min_partition_size=10, 
                                              self_balancing=True, seed=self.seed)
        return partitioner

    def load_federated_dataset(self):
        """
        Load the FederatedDataset and partition it according to the specified partition type.
        """
        self.logger.info("Loading federated dataset.")
        partitioner = self._initialize_partitioner()
        self.fds = FederatedDataset(dataset=self.datasetID, partitioners={"train": partitioner})
        self.client_datasets = [self.fds.load_partition(user_id) for user_id in range(self.num_clients)]
        self.logger.info("Federated dataset loaded and partitioned.")
        if self.label_flipping_attack:
            attackers = [cid for cid in range(self.num_attackers)] #manual setting attackers to be updated
            self.attack_client_datasets_structured(attackers , 1.00 ) #experiment to perfrom label flipping attack
            
        
        
        self._apply_transforms_to_clients()

    def _apply_transforms_to_clients(self):
        """
        Apply transformations to each client's dataset.
        """
        self.logger.info("Applying transformations to client datasets.")
        self.client_datasets = [client_dataset.with_transform(DatasetUtils.apply_transforms) 
                                for client_dataset in self.client_datasets]
        self.logger.info("Transformations applied to all client datasets.")

    def load_test_data(self):
        """
        Load and transform the test dataset.
        """
        self.logger.info("Loading test dataset.")
        test_dataset = self.fds.load_split("test")
        transformed_test = test_dataset.with_transform(DatasetUtils.apply_transforms)
        self.logger.info("Test dataset loaded and transformed.")
        return transformed_test
    
    
    def get_default_label_mapping(self, num_classes: int) -> dict:
        """
        Returns a default structured label mapping where each label is mapped to (label + 1) modulo num_classes.
        For example, with num_classes=10, the mapping is {0: 1, 1: 2, ..., 9: 0}.
        """
        return {i: (i + 1) % num_classes for i in range(num_classes)}

    def attack_client_datasets_structured(self, attacked_clients: list, flip_percentage: float, num_classes: int = 10, mapping: dict = None):
        """
        Attacks specified clients' datasets by flipping a given percentage of their labels using a structured mapping.
        If no mapping is provided, a default mapping {i: (i+1)%num_classes} is used.
        
        Args:
            attacked_clients (List[int]): List of client indices to attack.
            flip_percentage (float): Percentage (0-100) of labels to flip in each attacked client dataset.
            num_classes (int): Total number of classes.
            mapping (dict, optional): A mapping dict defining the new label for each original label.
        """
        if mapping is None:
            mapping = self.get_default_label_mapping(num_classes)
        self.logger.info(f"Attacking clients {attacked_clients} with {flip_percentage}% structured label flips.")
        for client_id in attacked_clients:
            self.logger.info(f"Attacking client {client_id} dataset using structured mapping.")
            client_dataset = self.client_datasets[client_id]
            total_examples = len(client_dataset)
            n_to_flip = int(total_examples * flip_percentage)
            # Randomly sample indices to flip
            indices_to_flip = set(random.sample(range(total_examples), n_to_flip))
            
            def structured_flip(example, idx):
                if idx in indices_to_flip:
                    original = example['label']
                    example['label'] = mapping.get(original, original)
                return example

            # Use with_indices=True to get the index passed to the function
            client_dataset = client_dataset.map(lambda example, idx: structured_flip(example, idx), with_indices=True)
            # Update the client dataset with the attacked version
            self.client_datasets[client_id] = client_dataset

        self.logger.info("Structured attack complete on selected client datasets.")


class FederatedDataLoader:
    def __init__(self, dataset_handler: DatasetHandler, batch_size: int , device: str):
        self.dataset_handler = dataset_handler
        self.batch_size = batch_size
        self.device = device

    def _get_client_data(self, client_id: int) -> DataLoader:
        return DataLoader(
            self.dataset_handler.client_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,          
            pin_memory=True if self.device == 'cuda' else False
        )

    def _get_test_data(self) -> DataLoader:
        test_dataset = self.dataset_handler.load_test_data()
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

def server_dataset(datasetID: str, batch_size: int, device: str) -> DataLoader:
        """
        Load the server dataset for FLTRUST
        """
        fds = FederatedDataset(dataset=datasetID, partitioners={"train": IidPartitioner(num_partitions=600)})
        server_dataset = fds.load_partition(0).with_transform(DatasetUtils.apply_transforms)
        return DataLoader(
            server_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
