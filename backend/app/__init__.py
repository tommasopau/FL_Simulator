from flask import Flask
from utils.logger import setup_logger
setup_logger()
from utils.db import get_engine, create_tables
from sqlalchemy import inspect
from .routes.simulation_bp import sim_bp
from .routes.configuration_bp import config_bp
from .routes.query_bp import query_bp


def create_app(config_overrides=None):
    # Instantiate Flask
    app = Flask(__name__)
    
    # Setup logger and other initial configurations
    setup_logger()
    
    # Initialize database
    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table("simulations_results"):
        create_tables(engine)
    
    # Register Blueprints
    app.register_blueprint(config_bp)
    app.register_blueprint(sim_bp)
    app.register_blueprint(query_bp)
    
    # You can add more configuration logic or pass configuration here if needed.
    
    return app

