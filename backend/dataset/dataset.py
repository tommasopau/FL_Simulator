import logging
from abc import ABC, abstractmethod
from backend.utils.constants import NORMALIZATION_PARAMS, TARGET_LABELS
import pandas as pd
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

##TESTINg
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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
        self.client_label_distributions = {}  #TESTING
        self.clusters = {}
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
            
        
        self._calculate_label_distributions()
        optimal_results = self.find_optimal_k_kmeans(max_k=8, plot=False)
        print(f"Optimal K: {optimal_results['optimal_k']}")
        clustering_results = self.perform_clustering_analysis(
            methods=['kmeans', 'dbscan', 'hierarchical'], 
            optimal_k=optimal_results['optimal_k'],
            plot=False
        )
        
        
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
    ### TESTING METHODS ###
    def _calculate_label_distributions(self):
        """
        Calculate the label distribution (percentages) for each client's dataset.
        """
        self.logger.info("Calculating label distributions for all clients.")
        self.client_label_distributions = {}
        
        for client_id in range(self.num_clients):
            client_dataset = self.client_datasets[client_id]
            
            # Extract labels based on dataset type
            if hasattr(client_dataset, 'with_format'):
                # For image datasets (HuggingFace datasets)
                labels = [example['label'] for example in client_dataset]
            else:
                # For tabular datasets (TensorDataset)
                labels = [int(client_dataset[i][1].item()) for i in range(len(client_dataset))]
            
            # Count labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            # Calculate percentages
            label_percentages = {}
            for label, count in zip(unique_labels, counts):
                label_percentages[int(label)] = (count / total_samples)
            
            self.client_label_distributions[client_id] = {
                'percentages': label_percentages,
            }
            self.logger.info(f"Client {client_id} label distribution: {label_percentages}")
        
    def get_label_distribution_matrix(self) -> np.ndarray:
        """
        Convert client label distributions to a matrix for clustering.
        
        Returns:
            np.ndarray: Matrix where each row represents a client and columns represent label percentages
        """
        if not self.client_label_distributions:
            self.logger.warning("Label distributions not calculated. Run _calculate_label_distributions() first.")
            return np.array([])
        
        # Get all unique labels across all clients
        all_labels = set()
        for client_dist in self.client_label_distributions.values():
            all_labels.update(client_dist['percentages'].keys())
        all_labels = sorted(list(all_labels))
        
        # Create matrix
        matrix = []
        for client_id in range(self.num_clients):
            client_dist = self.client_label_distributions[client_id]['percentages']
            row = [client_dist.get(label, 0.0) for label in all_labels]
            matrix.append(row)
        
        return np.array(matrix)
    def find_optimal_k_kmeans(self, max_k: int = 10, plot: bool = True) -> dict:
        """
        Find optimal number of clusters for K-means using silhouette score.
        
        Args:
            max_k (int): Maximum number of clusters to test
            plot (bool): Whether to plot the results
            
        Returns:
            dict: Results containing optimal k, scores, and clustering results
        """
        self.logger.info("Finding optimal K for K-means clustering using silhouette score.")
        
        X = self.get_label_distribution_matrix()
        if X.size == 0:
            self.logger.error("No label distribution data available.")
            return {}
        
        # Test different k values
        k_range = range(2, min(max_k + 1, self.num_clients))
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        results = {}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            sil_score = silhouette_score(X, cluster_labels)
            cal_score = calinski_harabasz_score(X, cluster_labels)
            db_score = davies_bouldin_score(X, cluster_labels)
            
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            davies_bouldin_scores.append(db_score)
            
            results[k] = {
                'silhouette_score': sil_score,
                'calinski_harabasz_score': cal_score,
                'davies_bouldin_score': db_score,
                'cluster_labels': cluster_labels,
                'kmeans_model': kmeans
            }
            
            self.logger.info(f"K={k}: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.3f}, Davies-Bouldin={db_score:.3f}")
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        
        
        return {
            'optimal_k': optimal_k,
            'optimal_silhouette_score': max(silhouette_scores),
            'all_results': results,
            'feature_matrix': X
        }
    def perform_clustering_analysis(self, methods: list = ['kmeans', 'dbscan', 'hierarchical'], 
                                  optimal_k: int = None, plot: bool = True, 
                                  store_method: str = 'kmeans') -> dict:
        """
        Perform clustering analysis using multiple algorithms.
        
        Args:
            methods (list): List of clustering methods to use
            optimal_k (int): Number of clusters for methods that require it
            plot (bool): Whether to plot results
            store_method (str): Which clustering method results to store in self.clusters
            
        Returns:
            dict: Clustering results for all methods
        """
        self.logger.info(f"Performing clustering analysis using methods: {methods}")
        
        X = self.get_label_distribution_matrix()
        if X.size == 0:
            self.logger.error("No label distribution data available.")
            return {}
        
        results = {}
        
        # Determine optimal k if not provided
        if optimal_k is None and 'kmeans' in methods:
            kmeans_results = self.find_optimal_k_kmeans(plot=False)
            optimal_k = kmeans_results.get('optimal_k', 3)
        
        # K-Means
        if 'kmeans' in methods:
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.seed, n_init=10)
            kmeans_labels = kmeans.fit_predict(X)
            results['kmeans'] = {
                'labels': kmeans_labels,
                'model': kmeans,
                'silhouette_score': silhouette_score(X, kmeans_labels),
                'n_clusters': optimal_k
            }
        
        # DBSCAN
        if 'dbscan' in methods:
            # Try different eps values to find reasonable clustering
            best_dbscan = None
            best_score = -1
            best_eps = 0.1
            
            for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                dbscan = DBSCAN(eps=eps, min_samples=2)
                dbscan_labels = dbscan.fit_predict(X)
                
                if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:  # Valid clustering
                    score = silhouette_score(X, dbscan_labels)
                    if score > best_score:
                        best_score = score
                        best_dbscan = dbscan
                        best_eps = eps
            
            if best_dbscan is not None:
                dbscan_labels = best_dbscan.fit_predict(X)
                results['dbscan'] = {
                    'labels': dbscan_labels,
                    'model': best_dbscan,
                    'silhouette_score': best_score,
                    'n_clusters': len(set(dbscan_labels)),
                    'eps': best_eps
                }
            else:
                self.logger.warning("DBSCAN could not find valid clustering with tested parameters.")
        
        # Hierarchical/Agglomerative Clustering
        if 'hierarchical' in methods:
            hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
            hierarchical_labels = hierarchical.fit_predict(X)
            results['hierarchical'] = {
                'labels': hierarchical_labels,
                'model': hierarchical,
                'silhouette_score': silhouette_score(X, hierarchical_labels),
                'n_clusters': optimal_k
            }
        
        # Store the selected clustering method results
        if store_method in results:
            self.cluster_method = store_method
            labels = results[store_method]['labels']
            self.clusters = {client_id: int(labels[client_id]) for client_id in range(self.num_clients)}
            self.logger.info(f"Stored {store_method} clustering results. Clusters: {self.clusters}")
        else:
            self.logger.warning(f"Store method '{store_method}' not found in results. Clusters not stored.")
        
        # Print summary
        self._print_clustering_summary(results)
        
        return results
    def _print_clustering_summary(self, results):
        """Print summary of clustering results."""
        print("\n" + "="*50)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*50)
        
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print(f"  Number of clusters: {result['n_clusters']}")
            print(f"  Silhouette score: {result['silhouette_score']:.3f}")
            
            # Show cluster composition
            unique_labels, counts = np.unique(result['labels'], return_counts=True)
            print(f"  Cluster sizes: {dict(zip(unique_labels, counts))}")
            
            # Show which clients are in each cluster
            print("  Client assignments:")
            for cluster_id in unique_labels:
                if cluster_id != -1:  # Skip noise points in DBSCAN
                    clients_in_cluster = [i for i, label in enumerate(result['labels']) if label == cluster_id]
                    print(f"    Cluster {cluster_id}: {clients_in_cluster}")
    
    




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

