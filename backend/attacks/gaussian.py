import torch 
import numpy as np
from typing import List


def gaussian_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Malicious attack in which each malicious client's update is replaced by a randomly 
    generated vector sampled from a Gaussian distribution with a mean of 0 and a variance of 200.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate (unused).
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates to generate.
        device (torch.device): Computation device.
    
    Returns:
        List[torch.Tensor]: The list of gradients with malicious updates applied.
    """
    # Standard deviation is sqrt(200) ~ 14.1421.
    std = torch.sqrt(torch.tensor(200.0)).item()
    for i in range(num_attackers_epoch):
        v[i] = torch.normal(mean=0.0, std=std, size=v[i].size()).to(device)
    return v

