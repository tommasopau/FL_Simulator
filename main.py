import yaml
import torch
import logging
from utils.seeding import set_deterministic_mode
from utils.logger import setup_logger
from utils.models import MNISTCNN, FCMNIST
from dataset import FederatedDataLoader, DatasetHandler
from server import Server, AggregationStrategy
from client.client import Client
import multiprocessing

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Initialize centralized logger
    setup_logger()  # Configures the root logger
    logger = logging.getLogger(__name__)
    logger.info("Main process started.")
    
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    config = load_config('config.yaml')
    federated_cfg = config['federated_learning']
    model_cfg = config['model']
    
    set_deterministic_mode(federated_cfg.get('seed', 42))
    
    # Verify device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Model selection based on config
    model_mapping = {
        'MNISTCNN': MNISTCNN,
        'FCMNIST': FCMNIST,
    }
    model_type = model_cfg.get('type', 'MNISTCNN')
    model_class = model_mapping.get(model_type)
    if model_class is None:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    global_model = model_class().to(device)
    logger.info(f"Initialized global model: {model_type}")
    
    # Initialize DatasetHandler
    federated_dataset_handler = DatasetHandler(
        datasetID=federated_cfg['dataset'],
        num_clients=federated_cfg['num_clients'],
        partition_type=federated_cfg['partition_type'],
        alpha=federated_cfg['alpha']
    )
    federated_dataset_handler._initialize_partitioner()
    federated_dataset_handler.load_federated_dataset()
    
    
    
    # Initialize FederatedDataLoader
    federated_data_loader = FederatedDataLoader(
        dataset_handler=federated_dataset_handler,
        batch_size=federated_cfg['batch_size'],
        device=device
    )
    
    aggregation_strategy = AggregationStrategy[federated_cfg['aggregation_strategy']]
    
    server = Server(
        federated_data_loader=federated_data_loader,
        global_model=global_model,
        aggregation_strategy=aggregation_strategy,
        global_epochs=federated_cfg['global_epochs'],
        local_epochs=federated_cfg['local_epochs'],
        learning_rate=federated_cfg['learning_rate'],
        batch_size=federated_cfg['batch_size'],
        device=device
    )
    
    server.run_federated_training(total_epochs=federated_cfg['global_epochs'], sampled_clients=federated_cfg['sampled_clients'])
    
    
    logger.info("Main process completed.")

if __name__ == '__main__':
    main()