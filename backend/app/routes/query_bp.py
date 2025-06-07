from flask import Blueprint, request, jsonify
import logging
from backend.app.services.query_services import filter_query

query_bp = Blueprint('query_bp', __name__)

@query_bp.route('/query', methods=['POST'])
def execute_query():
    logger = logging.getLogger(__name__)
    filters = request.json.get('filters', [])
    try:
        data = filter_query(filters)
        return jsonify({"result": data}), 200
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return jsonify({"error": str(e)}), 400
    