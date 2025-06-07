from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

class SimulationResult(db.Model):
    __tablename__ = 'simulation_results'
    id = db.Column(db.Integer, primary_key=True)
    dataset = db.Column(db.String, nullable=False)
    num_clients = db.Column(db.Integer, nullable=False)
    alpha = db.Column(db.Float, nullable=False)
    attack = db.Column(db.String, nullable=True)
    batch_size = db.Column(db.Integer, nullable=False)
    global_epochs = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    local_epochs = db.Column(db.Integer, nullable=False)
    num_attackers = db.Column(db.Integer, nullable=False)
    partition_type = db.Column(db.String, nullable=False)
    sampled_clients = db.Column(db.Integer, nullable=False)
    seed = db.Column(db.Integer, nullable=False)
    local_DP_SGD = db.Column(db.Boolean, nullable=False)
    aggregation_strategy = db.Column(db.String, nullable=False)
    model_type = db.Column(db.String, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)

