import os
from flask import Flask
from fl_simulator.utils.logger import setup_logger
from fl_simulator.db import db  # Flask‑SQLAlchemy instance
from .routes.simulation_bp import sim_bp
from .routes.query_bp import query_bp


def create_app(config_overrides=None):
    app = Flask(__name__)
    setup_logger()

    app.config.from_object('backend.app.app_config.Config')

    if config_overrides:
        app.config.from_mapping(config_overrides)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    app.register_blueprint(sim_bp)
    app.register_blueprint(query_bp)

    return app
