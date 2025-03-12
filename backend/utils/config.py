from pydantic import BaseModel, ValidationError, Field
from typing import Dict
import logging
import json

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    type: str = Field(..., description="Type of the model. Must be 'MNISTCNN' or 'FCMNIST'.")

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
