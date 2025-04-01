from sqlalchemy import and_
from utils.constants import ALLOWED_FILTERS
from utils.db import get_session, get_engine, SimulationResult

def filter_query(filters):
    engine = get_engine()
    session = get_session(engine)
    try:
        query = session.query(SimulationResult)
        filter_conditions = []
        for f in filters:
            field = f.get('field')
            operator = f.get('operator')
            value = f.get('value')
            
            if field not in ALLOWED_FILTERS:
                raise ValueError(f"Field '{field}' is not allowed.")
            if operator not in ALLOWED_FILTERS[field]:
                raise ValueError(f"Operator '{operator}' is not allowed for field '{field}'.")
            
            column = getattr(SimulationResult, field, None)
            if not column:
                raise ValueError(f"Column '{field}' not found in the model.")
            
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
                raise ValueError(f"Invalid operator '{operator}'.")
            
            filter_conditions.append(condition)
        
        if filter_conditions:
            query = query.filter(and_(*filter_conditions))
        
        results = query.all()
        data = [result.__dict__ for result in results]
        for item in data:
            item.pop('_sa_instance_state', None)
        
        return data
    finally:
        session.close()