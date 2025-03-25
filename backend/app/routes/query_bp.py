from flask import Blueprint, request, jsonify
import logging
from sqlalchemy import and_
from utils.constants import ALLOWED_FILTERS
from utils.db import get_session, get_engine , SimulationResult

query_bp = Blueprint('query_bp', __name__)

@query_bp.route('/query', methods=['POST'])
def execute_query():
    logger = logging.getLogger(__name__)
    filters = request.json.get('filters', [])
    engine = get_engine()
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
            
            column = getattr(SimulationResult, field, None)
            if not column:
                return jsonify({"error": f"Column '{field}' not found in the model."}), 400

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
            query = query.filter(and_(*filter_conditions))
        
        results = query.all()
        data = [result.__dict__ for result in results]
        for item in data:
            item.pop('_sa_instance_state', None)
        
        return jsonify({"result": data}), 200
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return jsonify({"error": str(e)}), 400
    finally:
        session.close()