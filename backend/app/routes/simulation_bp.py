from flask import Blueprint, jsonify, request
import logging
import ray
from backend.utils.config import serialize_config, validate_config
from backend.utils.seeding import set_deterministic_mode
from backend.app.services.simulation_services import (
    initialize_model_device,
    load_datasets,
    create_server,
    store_simulation_result
)

sim_bp = Blueprint('simulation_bp', __name__)

@sim_bp.route('/simulation', methods=['POST'])
def start_federated_learning():
    ray.init()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Main process started.")
        
        # Get configuration from the API payload
        config = request.json
        if not config:
            return jsonify({"error": "No configuration provided"}), 400
        
        #config = serialize_config(config_input)
        validate_config(config)
        federated_cfg = config['federated_learning']
        
        set_deterministic_mode(config.get('seed', 42)) 
        
        global_model, device = initialize_model_device(config)
        logger.info(f"Using device: {device}")
        
        fed_data_loader = load_datasets(federated_cfg, device)
        
        server = create_server(federated_cfg, global_model, device, fed_data_loader)
        
        # Run training and store result in DB
        accuracy = server.run_federated_training()
        
        store_simulation_result(
            config=config,
            federated_cfg=federated_cfg,
            accuracy=accuracy
        )
        
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