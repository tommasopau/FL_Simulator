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
    Applies a Gaussian attack on client gradients by replacing malicious updates.

    Each malicious client's gradient is replaced with a random vector sampled from a 
    Gaussian distribution with a mean of 0 and a variance of 200.

    Args:
        v (List[torch.Tensor]): List of client gradients.
        lr (float): Learning rate (unused).
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates to generate.
        device (torch.device): Computation device.

    Returns:
        List[torch.Tensor]: The modified list of gradients with malicious updates applied.
    """
    std = torch.sqrt(torch.tensor(200.0)).item()
    for i in range(num_attackers_epoch):
        v[i] = torch.normal(mean=0.0, std=std, size=v[i].size()).to(device)
    return v

