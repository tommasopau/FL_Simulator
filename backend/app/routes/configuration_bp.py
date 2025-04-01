from flask import Blueprint, request, jsonify
import logging
import yaml
from utils.config import validate_config
from app.services.configuration_services import save_config


config_bp = Blueprint('configuration_bp', __name__)

@config_bp.route('/configuration', methods=['POST'])
def update_config():
    logger = logging.getLogger(__name__)
    config_data = request.json
    try:
        validate_config(config_data)
        save_config(config_data)
        logger.info(f"Config updated: {config_data}")
        return jsonify({"status": "config updated"}), 200
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return jsonify({"error": str(e)}), 400
    
