import torch
import logging
from backend.utils.constants import MODEL_MAPPING
from backend.utils.config import Config ,validate_config
from backend.dataset.dataset import FederatedDataLoader, DatasetHandler
from backend.server.server import FLTrustServer, AttackServer, AggregationStrategy, AttackType
from backend.db import db, SimulationResult
from backend.utils.seeding import set_deterministic_mode


logger = logging.getLogger(__name__)
class SimulationService:
    def __init__(self, config: Config):
        logger.info("Initializing SimulationService")
        self.config = config
        self.device = self._setup_device()
        self.global_model = self._initialize_model()
        self.federated_data_loader = self._load_datasets()
        self.server = self._create_server()
        
        logger.info(f"Configuration: Model={config.model.type}, "
                   f"Dataset={config.dataset.dataset}, "
                   f"Clients={config.dataset.num_clients}, "
                   f"Attack={config.attack.attack}, "
                   f"Aggregation={config.aggregation.aggregation_strategy}")
    
    def _setup_device(self) -> str:
        """Setup computing device"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        return device
    
    def _initialize_model(self):
        """Initialize the global model"""
        model_class = MODEL_MAPPING.get(self.config.model.type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")
        
        
        model = model_class()
        
        return model.to(self.device)
    
    def _load_datasets(self) -> FederatedDataLoader:
        """Load and prepare federated datasets"""
        dataset_handler = DatasetHandler(
            datasetID=self.config.dataset.dataset,
            num_clients=self.config.dataset.num_clients,
            partition_type=self.config.dataset.partition_type.value,
            alpha=self.config.dataset.alpha
        )
        
        dataset_handler.load_federated_dataset()
        
        
        if self.config.attack.attack == AttackType.LABEL_FLIP:
            dataset_handler.label_flipping_attack = True
            dataset_handler.num_attackers = self.config.attack.num_attackers
        
        return FederatedDataLoader(
            dataset_handler=dataset_handler,
            batch_size=self.config.dataset.batch_size,
            device=self.device
        )
    
    def _create_server(self):
        """Create server instance based on configuration"""
        logger.info("Creating federated learning server")
        
        try:
            attack_type = self.config.attack.attack
            aggregation_strategy = self.config.aggregation.aggregation_strategy
            
            logger.info(f"Server configuration: Aggregation={aggregation_strategy}, Attack={attack_type}")
            
            if aggregation_strategy == AggregationStrategy.FLTRUST:
                logger.info("Creating FLTrustServer")
                
                server_data_loader = self.federated_data_loader.dataset_handler.server_dataset(
                    batch_size=self.config.dataset.batch_size
                )
                logger.info("Server dataset loaded for FLTrust")
                
                server = FLTrustServer(
                    server_data_loader=server_data_loader,
                    attack_type=attack_type,
                    federated_data_loader=self.federated_data_loader,
                    global_model=self.global_model,
                    aggregation_strategy=aggregation_strategy,
                    sampled=self.config.training.sampled_clients,
                    global_epochs=self.config.training.global_epochs,
                    local_epochs=self.config.training.local_epochs,
                    learning_rate=self.config.training.learning_rate,
                    batch_size=self.config.dataset.batch_size,
                    local_dp=self.config.training.local_DP_SGD,
                    fedprox=self.config.training.fedprox,
                    fedprox_mu=self.config.training.fedprox_mu,  
                    optimizer=self.config.training.optimizer,      
                    momentum=self.config.training.momentum,        
                    device=self.device,
                    f=self.config.attack.num_attackers
                )
            else:
                logger.info("Creating AttackServer")
                
                server = AttackServer(
                    attack_type=attack_type,
                    federated_data_loader=self.federated_data_loader,
                    global_model=self.global_model,
                    aggregation_strategy=aggregation_strategy,
                    sampled=self.config.training.sampled_clients,
                    global_epochs=self.config.training.global_epochs,
                    local_epochs=self.config.training.local_epochs,
                    learning_rate=self.config.training.learning_rate,
                    batch_size=self.config.dataset.batch_size,
                    local_dp=self.config.training.local_DP_SGD,
                    fedprox=self.config.training.fedprox,
                    fedprox_mu=self.config.training.fedprox_mu,  
                    optimizer=self.config.training.optimizer,      
                    momentum=self.config.training.momentum,        
                    device=self.device,
                    f=self.config.attack.num_attackers
                )
            
            logger.info("Server created successfully")
            
            
            return server
            
        except Exception as e:
            logger.error(f"Failed to create server: {str(e)}")
            raise
    
    def run_simulation(self) -> float:
        """Run the federated learning simulation"""
        
        if self.config.system.seed:
            set_deterministic_mode(self.config.system.seed)
        
        
        
        accuracy = self.server.run_federated_training()
        
        
        self._store_results(accuracy)
        
        return accuracy
    def _store_results(self, accuracy: float):
        """Store simulation results in database"""
        logger.debug("Preparing simulation results for database storage")
        
        try:
            result = SimulationResult(
                # Model parameters
                model_type=self.config.model.type,
                
                # Dataset parameters
                dataset=self.config.dataset.dataset,
                num_clients=self.config.dataset.num_clients,
                alpha=self.config.dataset.alpha,
                batch_size=self.config.dataset.batch_size,
                partition_type=self.config.dataset.partition_type.value,
                
                # Training parameters
                global_epochs=self.config.training.global_epochs,
                local_epochs=self.config.training.local_epochs,
                learning_rate=self.config.training.learning_rate,
                sampled_clients=self.config.training.sampled_clients,
                local_DP_SGD=self.config.training.local_DP_SGD,
                fedprox=self.config.training.fedprox,
                fedprox_mu=self.config.training.fedprox_mu,  
                optimizer=self.config.training.optimizer,      
                momentum=self.config.training.momentum,
                
                # Attack parameters
                attack=self.config.attack.attack.value,
                num_attackers=self.config.attack.num_attackers,
                
                # System parameters
                seed=self.config.system.seed,
                
                # Aggregation parameters
                aggregation_strategy=self.config.aggregation.aggregation_strategy.value,
                
                # Results
                accuracy=accuracy
                
            )
            
            db.session.add(result)
            db.session.commit()
            
            logger.info(f"Simulation result stored - ID: {result.id}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to store simulation results: {str(e)}")
            db.session.rollback()
            raise
def create_simulation_from_dict(config_dict: dict) -> SimulationService:
    """Create a simulation service from a dictionary configuration"""
    try:
        # Add validation to ensure config_dict is not None or empty
        if not config_dict:
            raise ValueError("Configuration dictionary cannot be None or empty")
        
        config = validate_config(config_dict)
        
        
        if config is None:
            raise ValueError("Configuration validation returned None")
        
        logger.info("Configuration validated successfully")
        
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        raise ValueError(f"Configuration validation error: {e}")
    
    return SimulationService(config)
'''
def initialize_model_device(config):
    
    
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
            fedprox = federated_cfg['fedprox'],
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
            fedprox=federated_cfg['fedprox'],
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
        fedprox=federated_cfg['fedprox'],
        aggregation_strategy=federated_cfg['aggregation_strategy'],
        model_type=config['model']['type'],
        accuracy=accuracy
    )
    db.session.add(result)
    db.session.commit()
    return 
'''