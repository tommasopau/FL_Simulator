import torch
import logging
import json
import ray
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Local imports with correct paths - updated to match current structure
from utils.seeding import set_deterministic_mode
from utils.logger import setup_logger
from utils.config import load_config, serialize_config, validate_config
from dataset.dataset import FederatedDataLoader, DatasetHandler, DatasetHandlerTab
from server.server import AttackServer, FLTrustServer, AggregationStrategy, AttackType
from utils.constants import MODEL_MAPPING
import collections


   
def get_client_class_counts(dataset_handler):
    """
    Returns a dictionary where each key is a client index (or ID)
    and the value is another dictionary containing the number of samples for each class.
    """
    client_counts = {}
    for i, client_ds in enumerate(dataset_handler.client_datasets):
        # Count the labels for this client
        counts = collections.Counter()
        for sample in client_ds:
            label = sample['label']
            counts[label] += 1
        client_counts[f"client_{i}"] = dict(counts)
    return client_counts


def main():
    # Initialize Ray
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Main process started.")
    ray.init()
    try:
        # Initialize centralized logger
        
        
        # Load and validate config first
        config = load_config('config.yaml')
        config = serialize_config(config)
        validate_config(config)
            
        federated_cfg = config['federated_learning']
        model_cfg = config['model']
        
        # Set deterministic mode after loading config
        set_deterministic_mode(federated_cfg.get('seed', 42))    
        
        # Verify device availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
            
        # Initialize model
        model_type = model_cfg.get('type', 'FCMNIST')
        model_class = MODEL_MAPPING.get(model_type, None)
        if model_class is None:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
            
        global_model = model_class().to(device)
        logger.info(f"Initialized global model: {model_type}")
            
        # Determine the appropriate DatasetHandler based on the dataset type
        dataset_id = federated_cfg['dataset'].lower()
        if dataset_id not in ['mnist', 'fashion_mnist', 'uoft-cs/cifar10']:
            dataset_handler_class = DatasetHandlerTab
        else:
            dataset_handler_class = DatasetHandler

        federated_dataset_handler = dataset_handler_class(
            datasetID=federated_cfg['dataset'],
            num_clients=federated_cfg['num_clients'],
            partition_type=federated_cfg['partition_type'],
            alpha=federated_cfg['alpha'],
            seed=federated_cfg.get('seed', 42),
        )
        
        # Handle label flipping attack at dataset level
        if federated_cfg['attack'].upper() == 'LABEL_FLIP':
            federated_dataset_handler.label_flipping_attack = True
            federated_dataset_handler.num_attackers = federated_cfg['num_attackers']
            
        federated_dataset_handler._initialize_partitioner()
        federated_dataset_handler.load_federated_dataset()    
        
        # Get and log the counts per client
        counts = get_client_class_counts(federated_dataset_handler)
        
        
        # Initialize FederatedDataLoader
        federated_data_loader = FederatedDataLoader(
            dataset_handler=federated_dataset_handler,
            batch_size=federated_cfg['batch_size'],
            device=device
        )
        
        # Get attack type
        attack_str = federated_cfg.get('attack', 'NO_ATTACK')
        attack_type = AttackType[attack_str.upper()]
        
        # Create server based on aggregation strategy
        if federated_cfg['aggregation_strategy'].upper() == 'FLTRUST':
            server_data_loader = federated_dataset_handler.server_dataset(
                batch_size=federated_cfg['batch_size']
            )
            # Use FLTrustServer (note: corrected class name)
            server = FLTrustServer(
                server_data_loader=server_data_loader,
                attack_type=attack_type,
                federated_data_loader=federated_data_loader,
                global_model=global_model,
                aggregation_strategy=AggregationStrategy.FLTRUST,
                sampled=federated_cfg['sampled_clients'],
                global_epochs=federated_cfg['global_epochs'],
                local_epochs=federated_cfg['local_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                batch_size=federated_cfg['batch_size'],
                local_dp=federated_cfg['local_DP_SGD'],
                device=device,
                f=federated_cfg['num_attackers'],
            )
        else:
            aggregation_strategy = AggregationStrategy[federated_cfg['aggregation_strategy']]
            server = AttackServer(
                attack_type=attack_type,
                federated_data_loader=federated_data_loader,
                global_model=global_model,
                aggregation_strategy=aggregation_strategy,
                sampled=federated_cfg['sampled_clients'],
                global_epochs=federated_cfg['global_epochs'],
                local_epochs=federated_cfg['local_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                batch_size=federated_cfg['batch_size'],
                local_dp=federated_cfg['local_DP_SGD'],
                device=device,
                f=federated_cfg['num_attackers'],
            )

        # Run federated training
        logger.info("Starting federated training simulation...")
        accuracy = server.run_federated_training()
        logger.info(f"Training completed with final accuracy: {accuracy}")
        
        # Save results
        try:
            result_filepath = "results.txt"
            with open(result_filepath, "a") as result_file:
                result_file.write("Simulation Result\n")
                result_file.write(f"Accuracy: {accuracy}\n")
                result_file.write("Configuration:\n")
                result_file.write(json.dumps(config, indent=4))
                result_file.write("\n" + "="*50 + "\n\n")
            logger.info(f"Results saved to {result_filepath}")
            
            # Save epoch-wise results if available
            if hasattr(server, 'epoch_results') and server.epoch_results:
                epoch_filepath = "simulation_epoch_results.txt"
                with open(epoch_filepath, "a") as epoch_file:
                    epoch_file.write("Simulation Epoch Results\n")
                    epoch_file.write("Configuration:\n")
                    epoch_file.write(json.dumps(config, indent=4))
                    epoch_file.write("\nEpoch Results:\n")
                    epoch_file.write(json.dumps(server.epoch_results))
                    epoch_file.write("\n" + "="*50 + "\n\n")
                logger.info(f"Epoch results saved to {epoch_filepath}")
                
        except Exception as write_exc:
            logger.error(f"Failed to write simulation results to file: {write_exc}")
            
    except Exception as e:
        logger.error(f"Error during federated learning simulation: {e}")
        raise
    finally:
        ray.shutdown()
        logger.info("Ray shutdown completed.")

if __name__ == '__main__':
    main()