from flask import Blueprint, jsonify, request
import ray
from backend.app.services.simulation_services import (
    create_simulation_from_dict
)
from pydantic import ValidationError

sim_bp = Blueprint('simulation_bp', __name__)

@sim_bp.route('/simulation', methods=['POST'])
def start_federated_learning():
    ray.init()
    try:
        config_dict = request.get_json()
        
        
        service = create_simulation_from_dict(config_dict)
        accuracy = service.run_simulation()
        
        return jsonify({
            'success': True,
            'accuracy': accuracy,
        })
    
    except ValidationError as e:
        return jsonify({
            'success': False, 
            'error': 'Configuration validation failed',
            'details': e.errors()
        }), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
        
    finally:
        ray.shutdown()