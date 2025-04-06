import logging
from abc import ABC, abstractmethod
from utils.constants import NORMALIZATION_PARAMS, TARGET_LABELS
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader , TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from datasets import load_dataset
from .flwr_wrapper import CustomFederatedDataset

class AbstractDatasetHandler(ABC):
    def __init__(self, datasetID: str, num_clients: int, partition_type: str, alpha: float, seed: int = 33):
        self.datasetID = datasetID
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fds = None
        self.client_datasets = None
        self.label_flipping_attack = False
        self.num_attackers = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetHandler initialized.")

    @abstractmethod
    def load_federated_dataset(self):
        pass

    @abstractmethod
    def load_test_data(self):
        pass
    
    


class DatasetHandler(AbstractDatasetHandler):
    def __init__(self, datasetID: str, num_clients: int, partition_type: str, alpha: float, seed: int = 33):
        super().__init__(datasetID, num_clients, partition_type, alpha, seed)
    
    @staticmethod
    def apply_transforms(batch , dataset_type: str) -> dict:
        """
        Apply transformations to the images in the dataset.
        """
        mean, std = NORMALIZATION_PARAMS.get(dataset_type, ((0.5,), (0.5,)))
        pytorch_transforms = Compose([ToTensor(), Normalize(mean , std)]) #Mnist Value -> add fashion_mnist (0.1307,), (0.3081,
        batch['image'] = [pytorch_transforms(img) for img in batch['image']]
        return batch
    
    


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
        transform = lambda batch: DatasetHandler.apply_transforms(batch, dataset_type=self.datasetID)
        self.client_datasets = [client_dataset.with_transform(transform) 
                                for client_dataset in self.client_datasets]
        self.logger.info("Transformations applied to all client datasets.")

    def load_test_data(self):
        """
        Load and transform the test dataset.
        """
        self.logger.info("Loading test dataset.")
        test_dataset = self.fds.load_split("test")
        transform = lambda batch: DatasetHandler.apply_transforms(batch, dataset_type=self.datasetID)
        transformed_test = test_dataset.with_transform(transform)
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
        
        
        
        
        
    def server_dataset(self, batch_size: int) -> DataLoader:
        fds = FederatedDataset(dataset=self.datasetID, partitioners={"train": IidPartitioner(num_partitions=600)})
        transform = lambda batch: DatasetHandler.apply_transforms(batch, dataset_type=self.datasetID)
        server_dataset = fds.load_partition(0).with_transform(transform)
        return DataLoader(
            server_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )




#-----Tabular Data Handling-----

class DatasetHandlerTab(AbstractDatasetHandler):
    def __init__(self, datasetID: str, num_clients: int, partition_type: str, alpha: float, seed: int = 33):
        super().__init__(datasetID, num_clients, partition_type, alpha, seed)
    """
    DatasetHandlerTab extends DatasetHandler to handle tabular datasets.
    It overrides methods that require adjustments for tabular data processing.
    """
    @staticmethod
    def transform_client_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to the client's dataset.
        """
        dataset.dropna(inplace=True)
        
        dataset.drop_duplicates(inplace=True, ignore_index=True)
        
        
            
        
        categorical_columns = dataset.select_dtypes(include=['object','bool']).columns.tolist()
        if categorical_columns:
            ordinal_encoder =  OrdinalEncoder()
            dataset[categorical_columns] = ordinal_encoder.fit_transform(dataset[categorical_columns])
        
        
        label = dataset.columns[-1]
        
        X = dataset.drop(columns=[label],axis=1)
        
        y = dataset[label]
        
        #no local test set 
        X_train, y_train = X, y
        
        numeric_features = X_train.select_dtypes(include=['float64','float32' , 'int64']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        
        preprocess = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
        
        X_train = preprocess.fit_transform(X_train)
       
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).view(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        return train_dataset
        
    def _initialize_partitioner(self):
        """
        Initialize the partitioner based on the partition type.
        """
        label = TARGET_LABELS.get(self.datasetID, 'label')
        if self.partition_type == 'iid':
            partitioner = IidPartitioner(num_partitions=self.num_clients)
        else:
            partitioner = DirichletPartitioner(num_partitions=self.num_clients, partition_by=label, 
                                              alpha=self.alpha, min_partition_size=10, 
                                              self_balancing=True, seed=self.seed)
        return partitioner

    def load_federated_dataset(self):
        """
        Load the FederatedDataset and partition it according to the specified partition type.
        """
        self.logger.info("Loading federated dataset.")
        partitioner = self._initialize_partitioner()
        if self.datasetID == 'mstz/covertype':
            self.fds = FederatedDataset(dataset=self.datasetID, subset='covertype', partitioners={"train": partitioner})
        elif self.datasetID == 'mstz/kddcup':
            dataset = load_dataset(self.datasetID, split='train')
            
            dataset = dataset.shuffle(seed=88).select(range(800000))
            
            
            logging.info(f"Loaded {len(dataset)} rows from the dataset.")
            self.fds = CustomFederatedDataset(dataset={'train': dataset}, partitioners={"train": partitioner})
        
        else:
            self.fds = FederatedDataset(dataset=self.datasetID, partitioners={"train": partitioner})
        self.client_datasets = [self.fds.load_partition(user_id).with_format('pandas')[:] for user_id in range(self.num_clients)]
        self.logger.info("Federated dataset loaded and partitioned.")
        
        
        
        if self.label_flipping_attack:
            attackers = [cid for cid in range(self.num_attackers)] #manual setting attackers to be updated
            # TODO: Implement label flipping attack for security testing.
            pass
        
        self._transform_clients_datasets()
        
    
    def _transform_clients_datasets(self):
        """
        Apply transformations to each client's dataset.
        """
        self.logger.info("Applying transformations to client datasets.")
        self.client_datasets = [self.transform_client_dataset(client_dataset) 
                                for client_dataset in self.client_datasets]
        self.logger.info("Transformations applied to all client datasets.")
        
    def load_test_data(self):
        """
        Load and transform the test dataset for tabular data by aggregating all training partitions and splitting.
        """
        self.logger.info("Loading test dataset for tabular data by aggregating all partitions and splitting.")
        # Aggregate the full dataset from all client partitions.
        
        full_dataset = pd.concat(
            [self.fds.load_partition(user_id).with_format('pandas')[:] for user_id in range(self.num_clients)],
            ignore_index=True
        )
        train_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=self.seed)
        
        # Process the test split
        test_tensor_dataset = DatasetHandlerTab.transform_client_dataset(test_data)
        
        self.logger.info("Test dataset loaded and transformed for tabular data.")
        return test_tensor_dataset

    def server_dataset(self, batch_size: int) -> DataLoader:
        """
        Load the server dataset for FLTRUST
        """
        
        fds = FederatedDataset(dataset=self.datasetID, partitioners={"train": IidPartitioner(num_partitions=600)})
        df = fds.load_partition(0).with_format('pandas')[:]
        # Use the static method from DatasetHandlerTab
        transformed_dataset = DatasetHandlerTab.transform_client_dataset(df)
            
        return DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )





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


#----SERVER FLTRUST 

def server_dataset(datasetID: str, batch_size: int, device: str) -> DataLoader:
        """
        Load the server dataset for FLTRUST
        """
        if datasetID not in ['mnist', 'cifar10', 'fashion_mnist']:
            fds = FederatedDataset(dataset=datasetID, partitioners={"train": IidPartitioner(num_partitions=600)})
            df = fds.load_partition(0).with_format('pandas')[:]
            # Use the static method from DatasetHandlerTab
            transformed_dataset = DatasetHandlerTab.transform_client_dataset(df)
            
            return DataLoader(
                transformed_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )
            
            
        else:
            fds = FederatedDataset(dataset=datasetID, partitioners={"train": IidPartitioner(num_partitions=600)})
            transform = lambda batch: DatasetUtils.apply_transforms(batch, dataset_type=datasetID)
            server_dataset = fds.load_partition(0).with_transform(transform)
            return DataLoader(
                server_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )