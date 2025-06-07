import torch
from backend.utils.constants import MODEL_MAPPING

def initialize_model_device(config):
    # Setup device and model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = config['model'].get('type')
    model_class = MODEL_MAPPING.get(model_type , None)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    global_model = model_class().to(device)
    
    return global_model, device



from backend.dataset.dataset import FederatedDataLoader, DatasetHandler

def load_datasets(federated_cfg, device):
    dataset_handler = DatasetHandler(
        datasetID=federated_cfg['dataset'],
        num_clients=federated_cfg['num_clients'],
        partition_type=federated_cfg['partition_type'],
        alpha=federated_cfg['alpha']
    )
    
    dataset_handler.load_federated_dataset()
    
    if federated_cfg['attack'].upper() == 'LABEL_FLIP':
        dataset_handler.label_flipping_attack = True
        dataset_handler.num_attackers = federated_cfg['num_attackers']
        
    federated_data_loader = FederatedDataLoader(
        dataset_handler=dataset_handler,
        batch_size=federated_cfg['batch_size'],
        device=device
    )
    
    
    return federated_data_loader 


from backend.server.server import FLTrustServer, AttackServer, AggregationStrategy, AttackType

def create_server(federated_cfg, global_model, device, fed_data_loader):
    """
    Creates and returns the server instance based on the configuration.
    """
    attack_type = AttackType[federated_cfg['attack'].upper()]

    if federated_cfg['aggregation_strategy'].upper() == 'FLTRUST':
        server_data_loader = fed_data_loader.dataset_handler.server_dataset(batch_size=federated_cfg['batch_size'])
        server = FLTrustServer(
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
    return server



from backend.db import db, SimulationResult

def store_simulation_result(config, federated_cfg, accuracy):
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
    db.session.add(result)
    db.session.commit()
    return 