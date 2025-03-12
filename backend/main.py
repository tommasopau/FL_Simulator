import torch
import logging
from utils.seeding import set_deterministic_mode
from utils.logger import setup_logger
setup_logger() 
from pydantic import ValidationError
from utils.models import MNISTCNN, FCMNIST , ZalandoCNN , FNet
from utils.constants import ALLOWED_FILTERS
from dataset import FederatedDataLoader, DatasetHandler , server_dataset
from server import Server, AggregationStrategy, AttackServer, AttackType , FLTrust
import multiprocessing
import ray
from flask import Flask, jsonify, request
from utils.config import Config, load_config , serialize_config , validate_config
from utils.db import get_engine, create_tables, get_session, SimulationResult
from sqlalchemy import and_, or_, Boolean , inspect


app = Flask(__name__)


# Initialize Database
engine = get_engine()
session = get_session(engine)

CONFIG_OVERRIDES = {}  # Keep updated parameters in memory

@app.route('/api/config', methods=['POST'])
def update_config():
    logger = logging.getLogger(__name__)
    global CONFIG_OVERRIDES
    CONFIG_OVERRIDES = request.json
    validate_config(CONFIG_OVERRIDES)
    logger.info(f"Config updated: {CONFIG_OVERRIDES}")
    return jsonify({"status": "config updated"}), 200


@app.route('/start', methods=['GET'])
def start_federated_learning():
    

    ray.init()
    try:
    
    
        # Initialize centralized logger
        
        logger = logging.getLogger(__name__)
        logger.info("Main process started.")
        
        config = load_config('config.yaml')
        config = config.model_dump()

        # Apply overrides
        for section, params in CONFIG_OVERRIDES.items():
            if section in config:
                config[section].update(params)

        validate_config(config)
        
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
            'ZalandoCNN' : ZalandoCNN ,
            'FNet' : FNet
        }
        model_type = model_cfg.get('type', None)
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
        if federated_cfg['attack'].upper() == 'LABEL_FLIP':
            federated_dataset_handler.label_flipping_attack = True
            federated_dataset_handler.num_attackers = federated_cfg['num_attackers']
        
        
        # Initialize FederatedDataLoader
        federated_data_loader = FederatedDataLoader(
            dataset_handler=federated_dataset_handler,
            batch_size=federated_cfg['batch_size'],
            device=device
        )
        attack_str = federated_cfg.get('attack', None)
        
        attack_type = AttackType[attack_str.upper()]
        
        if federated_cfg['aggregation_strategy'].upper() == 'FLTRUST':
            server_data_loader = server_dataset(federated_cfg['dataset'], federated_cfg['batch_size'], device)
            # Instantiate FLTrust server.
            server = FLTrust(
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
                sampled = federated_cfg['sampled_clients'],
                global_epochs=federated_cfg['global_epochs'],
                local_epochs=federated_cfg['local_epochs'],
                learning_rate=federated_cfg['learning_rate'],
                batch_size=federated_cfg['batch_size'],
                local_dp = federated_cfg['local_DP_SGD'],
                device=device,
                f = federated_cfg['num_attackers'],
                
                )
            

        # Run federated training and emit updates
        accuracy = server.run_federated_training()
        # Store result in DB
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
            model_type=model_cfg['type'],
            accuracy=accuracy
        )
        session.add(result)
        session.commit()
        
        status = "Training completed successfully."

        response = {
            "status": status,
            "accuracy": accuracy
        }

        if config['federated_learning']['aggregation_strategy'] == "KeTS":
            response["trustscores"] = server.trust_scores

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error during federated learning: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        ray.shutdown()
        


@app.route('/api/query', methods=['POST'])
def execute_query():
    logger = logging.getLogger(__name__)
    filters = request.json.get('filters', [])
    session = get_session(engine)
    try:
        query = session.query(SimulationResult)
        filter_conditions = []

        for f in filters:
            field = f.get('field')
            operator = f.get('operator')
            value = f.get('value')

            logger.info(f"Applying filter: {field} {operator} {value}")

            if field not in ALLOWED_FILTERS:
                return jsonify({"error": f"Field '{field}' is not allowed."}), 400
            if operator not in ALLOWED_FILTERS[field]:
                return jsonify({"error": f"Operator '{operator}' is not allowed for field '{field}'."}), 400
            logger.info(f"Filter: {field} {operator} {value}")
            # Access the column dynamically without the table prefix
            column = getattr(SimulationResult, field, None)
            logger.info(f"errore {column}")
            if not column:
                return jsonify({"error": f"Column '{field}' not found in the model."}), 400

            
            # Apply the operator
            if operator == "eq":
                condition = column == value
            elif operator == "gt":
                condition = column > value
            elif operator == "lt":
                condition = column < value
            elif operator == "gte":
                condition = column >= value
            elif operator == "lte":
                condition = column <= value
            else:
                return jsonify({"error": f"Invalid operator '{operator}'."}), 400
            
            filter_conditions.append(condition)

        if filter_conditions:
            logger.info(f"Filter conditions: {filter_conditions}")
            query = query.filter(and_(*filter_conditions))
        # Log the generated SQL query
        compiled_query = query.statement.compile(compile_kwargs={"literal_binds": True})
        logger.info(f"Generated SQL Query: {compiled_query}")
        results = query.all()
        data = [result.__dict__ for result in results]
        for item in data:
            item.pop('_sa_instance_state', None)  # Remove SQLAlchemy internals

        return jsonify({"result": data}), 200
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return jsonify({"error": str(e)}), 400
    finally:
        session.close()

@app.route('/')
def index():
    return "Federated Learning API is running."

# Update the main entry point to run with SocketIO
if __name__ == '__main__':
    inspector = inspect(engine)
    if not inspector.has_table("simulations_results"):
        create_tables(engine)
    app.run(host='0.0.0.0', port=8000)
