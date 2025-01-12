import logging
from utils.logger import setup_logger

import pandas as pd
from sklearn.model_selection import train_test_split

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
        pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
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
        
        self.apply_transforms_to_clients()

    def apply_transforms_to_clients(self):
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


class FederatedDataLoader:
    def __init__(self, dataset_handler: DatasetHandler, batch_size: int , device: str):
        self.dataset_handler = dataset_handler
        self.batch_size = batch_size
        self.device = device

    def get_client_data(self, client_id: int) -> DataLoader:
        return DataLoader(
            self.dataset_handler.client_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,          
            pin_memory=True if self.device == 'cuda' else False
        )

    def get_test_data(self) -> DataLoader:
        test_dataset = self.dataset_handler.load_test_data()
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
