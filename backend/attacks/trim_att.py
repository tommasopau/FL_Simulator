import torch 
import numpy as np
from typing import List


def trim_attack(
    v: List[torch.Tensor],
    net: torch.nn.Module,
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Local model poisoning attack using the trim method.
    
    Parameters:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
    """
    vi_shape = tuple(v[0].size())
    v_tran = torch.cat(v, dim=1)
    maximum_dim, _ = torch.max(v_tran, dim=1, keepdim=True)
    minimum_dim, _ = torch.min(v_tran, dim=1, keepdim=True)
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range( num_attackers_epoch):
        random_12 = (1. + torch.rand(*vi_shape)).to(device)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v