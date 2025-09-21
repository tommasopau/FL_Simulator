import os
import random
import torch
import numpy as np

def set_deterministic_mode(seed: int = 42):
    """Sets the deterministic mode and fixes the random seed for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # You can also try ":16:8"

    # Enable deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Fixing the random seed for reproducibility
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
