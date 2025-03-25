ALLOWED_FILTERS = {
    # Numeric Fields
    "global_epochs": ["eq", "gt", "lt", "gte", "lte"],
    "learning_rate": ["eq", "gt", "lt", "gte", "lte"],
    "num_clients": ["eq", "gt", "lt", "gte", "lte"],
    "alpha": ["eq", "gt", "lt", "gte", "lte"],
    "batch_size": ["eq", "gt", "lt", "gte", "lte"],
    "local_epochs": ["eq", "gt", "lt", "gte", "lte"],
    "num_attackers": ["eq", "gt", "lt", "gte", "lte"],
    "sampled_clients": ["eq", "gt", "lt", "gte", "lte"],
    "seed": ["eq"],
    "accuracy": ["eq", "gt", "lt", "gte", "lte"],

    # String Fields
    "dataset": ["eq"],
    "attack": ["eq"],
    "aggregation_strategy": ["eq"],
    "partition_type": ["eq"],
    "model_type": ["eq"],

    # Boolean Fields
    "local_DP_SGD": ["eq"],
    
    
}

NORMALIZATION_PARAMS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    # Add additional dataset normalization parameters here.
}
