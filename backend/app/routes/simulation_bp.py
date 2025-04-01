from flask import Blueprint, jsonify
import torch
import logging
import ray
from utils.models import MNISTCNN, FCMNIST, ZalandoCNN, FNet
from utils.constants import ALLOWED_FILTERS
from utils.config import load_config
from utils.db import get_session,get_engine , SimulationResult
from dataset.dataset import FederatedDataLoader, DatasetHandler, server_dataset
from server import FLTrust, AttackServer, AggregationStrategy, AttackType
from utils.seeding import set_deterministic_mode

sim_bp = Blueprint('simulation_bp', __name__)

@sim_bp.route('/simulation', methods=['GET'])
def start_federated_learning():
    ray.init()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Main process started.")
        config = load_config('config.yaml').model_dump()
        
                
        # Setup device and model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_deterministic_mode(config.get('seed', 42))
        logger.info(f"Using device: {device}")
        # Choose model based on configuration
        model_mapping = {
            'MNISTCNN': MNISTCNN,
            'FCMNIST': FCMNIST,
            'ZalandoCNN': ZalandoCNN,
            'FNet': FNet
        }
        model_type = config['model'].get('type')
        model_class = model_mapping.get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        global_model = model_class().to(device)
        logger.info(f"Initialized global model: {model_type}")
        
        # Initialize dataset handler and data loaders
        federated_cfg = config['federated_learning']
        dataset_handler = DatasetHandler(
            datasetID=federated_cfg['dataset'],
            num_clients=federated_cfg['num_clients'],
            partition_type=federated_cfg['partition_type'],
            alpha=federated_cfg['alpha']
        )
        dataset_handler._initialize_partitioner()
        dataset_handler.load_federated_dataset()
        if federated_cfg['attack'].upper() == 'LABEL_FLIP':
            dataset_handler.label_flipping_attack = True
            dataset_handler.num_attackers = federated_cfg['num_attackers']
        
        fed_data_loader = FederatedDataLoader(
            dataset_handler=dataset_handler,
            batch_size=federated_cfg['batch_size'],
            device=device
        )
        
        # Instantiate either FLTrust or AttackServer
        attack_type = AttackType[federated_cfg['attack'].upper()]
        if federated_cfg['aggregation_strategy'].upper() == 'FLTRUST':
            server_data_loader = dataset_handler.server_dataset(batch_size=federated_cfg['batch_size'])
            server = FLTrust(
                server_data_loader=server_data_loader,
                attack_type=attack_type,
                federated_data_loader=fed_data_loader,
                global_model=global_model,
                aggregation_strategy=AggregationStrategy.FLTRUST,
                sampled=federated_cfg['sampled_clients'],
                global_epochs=federated_cfg['global_epochs'],
                local_epochs=federated_cfg['local_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                batch_size=federated_cfg['batch_size'],
                local_dp=federated_cfg['local_DP_SGD'],
                device=device,
                f=federated_cfg['num_attackers']
            )
        else:
            aggregation_strategy = AggregationStrategy[federated_cfg['aggregation_strategy']]
            server = AttackServer(
                attack_type=attack_type,
                federated_data_loader=fed_data_loader,
                global_model=global_model,
                aggregation_strategy=aggregation_strategy,
                sampled=federated_cfg['sampled_clients'],
                global_epochs=federated_cfg['global_epochs'],
                local_epochs=federated_cfg['local_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                batch_size=federated_cfg['batch_size'],
                local_dp=federated_cfg['local_DP_SGD'],
                device=device,
                f=federated_cfg['num_attackers']
            )
        
        # Run training and store result in DB
        accuracy = server.run_federated_training()
        engine  = get_engine()
        session = get_session(engine)
        result = SimulationResult(
                dataset=federated_cfg['dataset'],
                num_clients=federated_cfg['num_clients'],
                alpha=federated_cfg['alpha'],
                attack=federated_cfg.get('attack'),
                batch_size=federated_cfg['batch_size'],
                global_epochs=federated_cfg['global_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                local_epochs=federated_cfg['local_epochs'],
                num_attackers=federated_cfg['num_attackers'],
                partition_type=federated_cfg['partition_type'],
                sampled_clients=federated_cfg['sampled_clients'],
                seed=federated_cfg['seed'],
                local_DP_SGD=federated_cfg['local_DP_SGD'],
                aggregation_strategy=federated_cfg['aggregation_strategy'],
                model_type=config['model']['type'],
                accuracy=accuracy
        )
        session.add(result)
        session.commit()
        resp = {
            "status": "Training completed successfully.",
            "accuracy": accuracy
        }
        if config['federated_learning']['aggregation_strategy'] == "KeTS":
            resp["trustscores"] = server.trust_scores

        return jsonify(resp)
    except Exception as e:
        logger.error(f"Error during federated learning: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        ray.shutdown()