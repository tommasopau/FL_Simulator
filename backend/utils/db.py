import os
from sqlalchemy import Column, Integer, Float, Text, String, Boolean , create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class SimulationResult(Base):
    __tablename__ = 'simulation_results'
    id = Column(Integer, primary_key=True)
    dataset = Column(String, nullable=False)
    num_clients = Column(Integer, nullable=False)
    alpha = Column(Float, nullable=False)
    attack = Column(String, nullable=True)
    batch_size = Column(Integer, nullable=False)
    global_epochs = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    local_epochs = Column(Integer, nullable=False)
    num_attackers = Column(Integer, nullable=False)
    partition_type = Column(String, nullable=False)
    sampled_clients = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    local_DP_SGD = Column(Boolean, nullable=False)
    aggregation_strategy = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)

def get_engine(db_path: str = None) -> create_engine:
    """
    Creates a SQLAlchemy engine connected to the specified database path.
    
    Args:
        db_path (str, optional): The database URL. If None, defaults to <project_root>/simulation.db.
    
    Returns:
        create_engine: A SQLAlchemy Engine instance.
    """
    if db_path is None:
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_file = os.path.join(project_root, 'simulation.db')
    else:
        db_file = db_path.replace('sqlite:///', '')

    db_uri = f"sqlite:///{db_file}"
    
    return create_engine(db_uri)

def create_tables(engine: create_engine) -> None:
    Base.metadata.create_all(engine)

def get_session(engine: create_engine):
    Session = sessionmaker(bind=engine)
    return Session()