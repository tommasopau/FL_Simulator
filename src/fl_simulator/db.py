from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

class SimulationResult(db.Model):
    __tablename__ = 'simulation_results'
    id = db.Column(db.Integer, primary_key=True)
    
    # Model Configuration
    model_type = db.Column(db.String, nullable=False)
    
    # Dataset Configuration
    dataset = db.Column(db.String, nullable=False)
    num_clients = db.Column(db.Integer, nullable=False)
    alpha = db.Column(db.Float, nullable=False)
    batch_size = db.Column(db.Integer, nullable=False)
    partition_type = db.Column(db.String, nullable=False)
    
    # Training Configuration
    global_epochs = db.Column(db.Integer, nullable=False)
    local_epochs = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    sampled_clients = db.Column(db.Integer, nullable=False)
    local_DP_SGD = db.Column(db.Boolean, nullable=False)
    fedprox = db.Column(db.Boolean, nullable=False)
    fedprox_mu = db.Column(db.Float, nullable=False)
    optimizer = db.Column(db.String, nullable=False)
    momentum = db.Column(db.Float, nullable=False)
    
    # Attack Configuration
    attack = db.Column(db.String, nullable=False)
    num_attackers = db.Column(db.Integer, nullable=True)
    # System Configuration
    seed = db.Column(db.Integer, nullable=False)
    
    
    # Aggregation Configuration
    aggregation_strategy = db.Column(db.String, nullable=False)
    
    # Results
    accuracy = db.Column(db.Float, nullable=False)

    

