import torch 
import numpy as np
from typing import List


def no_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
    ) -> List[torch.Tensor]:
    """
    No attack.
    
    Parameters:
        v (List[torch.Tensor]): List of gradients.
        
    """
    return v