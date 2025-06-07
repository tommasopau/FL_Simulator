import torch 
import numpy as np
from typing import List


def label_flip_attack (v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
    ) -> List[torch.Tensor]:
    """
    Label flip attack. It flips the labels of the first f clients but it happens before the training.
    Hence here we do not need to change the gradients.
    """
    return v