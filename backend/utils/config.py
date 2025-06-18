from pydantic import BaseModel, ValidationError, field_validator , Field, model_validator
from typing import Dict, Any
import logging
from enum import Enum
import json
from backend.server.server import AttackType , AggregationStrategy 

logger = logging.getLogger(__name__)
class PartitionType(Enum):
    IID = 'iid'
    NON_IID = 'non_iid'

class ModelConfig(BaseModel):
    type: str = Field(..., description="Type of the model. Must be 'MNISTCNN' or 'FCMNIST'.")
    
    @field_validator('type')
    def validate_model_type(cls, v):
        allowed_types = ['MNISTCNN', 'FCMNIST', 'ZalandoCNN', 'ResNetCIFAR10','TinyVGG','KDDSimpleNN']  # Add more as needed
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v

class DatasetConfig(BaseModel):
    dataset: str = Field(..., min_length=1, description="Dataset identifier.")
    num_clients: int = Field(..., ge=2, description="Number of federated clients. Must be at least 2.")
    alpha: float = Field(..., ge=0.05, le=10.0, description="Alpha value for the partitioning of the dataset.")
    batch_size: int = Field(..., ge=1, description="Batch size for the federated learning.")
    partition_type: PartitionType = Field(..., description="Type of data partitioning.")

class TrainingConfig(BaseModel):
    global_epochs: int = Field(..., ge=1, description="Number of global epochs for federated learning.")
    local_epochs: int = Field(..., ge=1, description="Number of local epochs for federated learning.")
    learning_rate: float = Field(..., gt=0, description="Learning rate for the federated learning.")
    sampled_clients: int = Field(..., ge=1, description="Number of clients sampled in each round.")
    
    local_DP_SGD: bool = Field(False, description="Flag to indicate if local DP-SGD is used.")
    fedprox: bool = Field(False, description="Whether to use FedProx algorithm")
    fedprox_mu: float = Field(0.01, ge=0, description="FedProx regularization parameter")
    
    optimizer: str = Field("SGD", description="Optimizer type")
    momentum: float = Field(0.9, ge=0, le=1, description="Momentum for SGD optimizer")

class AttackConfig(BaseModel):
    attack: AttackType = Field(AttackType.NO_ATTACK, description="Type of attack to be used")
    num_attackers: int = Field(0, ge=0, description="Number of attackers in the federated learning.")

class SystemConfig(BaseModel):
    seed: int = Field(None, description="Random seed for reproducibility.")

class Aggregation(BaseModel):
    aggregation_strategy:  AggregationStrategy = Field(AggregationStrategy.FEDAVG, description="Strategy for aggregating the models.")

class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    attack: AttackConfig = Field(default_factory=AttackConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    aggregation : Aggregation = Field(default_factory=Aggregation)
    
    @model_validator(mode='after')
    def validate_client_constraints(self) -> 'Config':
        # Validate sampled_clients doesn't exceed num_clients
        if self.training.sampled_clients > self.dataset.num_clients:
            raise ValueError("sampled_clients cannot exceed num_clients")
        
        # Validate attackers constraints
        if self.attack.attack != AttackType.NO_ATTACK:
            if self.attack.num_attackers <= 0:
                raise ValueError("num_attackers must be greater than 0 when an attack is specified")
            if self.attack.num_attackers > self.dataset.num_clients:
                raise ValueError("num_attackers cannot exceed num_clients")
        
        return self
    
def validate_config(overrides):
    try:
        return Config(**overrides)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ValidationError(f"Configuration validation error: {e}")

def load_config(config_path: str) -> Config:
    import yaml
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    try:
        return Config(**config_dict)
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        raise


def serialize_config(config: Config) -> dict:
    """
    Converts the Config model to a dictionary.
    """
    return config.model_dump()


'''
class FederatedConfig(BaseModel):
    dataset: str = Field(..., min_length=1, description="Dataset identifier.")
    num_clients: int = Field(..., ge=2, description="Number of federated clients. Must be at least 2.")
    alpha: float = Field(..., ge=0.05, le=10.0, description="Alpha value for the partitioning of the dataset.")
    attack: str = Field(None, description="Type of attack to be used")
    batch_size: int = Field(..., ge=1, description="Batch size for the federated learning.")
    global_epochs: int = Field(..., ge=1, description="Number of global epochs for federated learning.")
    learning_rate: float = Field(..., gt=0, description="Learning rate for the federated learning.")
    local_epochs: int = Field(..., ge=1, description="Number of local epochs for federated learning.")
    num_attackers: int = Field(..., ge=0, description="Number of attackers in the federated learning.")
    partition_type: str = Field(..., description="Type of data partitioning.")
    sampled_clients: int = Field(..., ge=1, description="Number of clients sampled in each round.")
    seed: int = Field(..., description="Random seed for reproducibility.")
    local_DP_SGD: bool = Field(..., description="Flag to indicate if local DP-SGD is used.")
    aggregation_strategy: str = Field(..., description="Strategy for aggregating the models.")
class Config(BaseModel):
    model: ModelConfig
    federated_learning: FederatedConfig
    

def load_config(config_path: str) -> Config:
    import yaml
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    try:
        return Config(**config_dict)
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        raise
def validate_config(overrides):
    try:
        Config(**overrides)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ValidationError(f"Configuration validation error: {e}")

def serialize_config(config: Config) -> dict:
    """
    Converts the Config model to a dictionary.
    """
    return config.model_dump()
'''