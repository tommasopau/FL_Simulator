import torch 
import numpy as np
from typing import List

def min_max_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Performs a local model poisoning attack using the min-max algorithm.

    Based on https://par.nsf.gov/servlets/purl/10286354 and its associated repository 
    (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning), this attack computes an optimal 
    scaling factor (gamma) to generate a malicious update that minimizes the maximum deviation 
    from benign updates. The malicious update then replaces the gradients for the first 
    num_attackers_epoch clients.

    Args:
        v (List[torch.Tensor]): List of client gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Computation device.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.

    Returns:
        List[torch.Tensor]: The modified list of gradients with the attack applied.
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    if dev_type == 'unit_vec':
        deviation = grad_mean / torch.norm(grad_mean, p=2)  # Unit vector (opposite to clean direction)
    elif dev_type == 'sign':
        deviation = torch.sign(grad_mean)  # Sign of the gradients
    elif dev_type == 'std':
        deviation = torch.std(catv, dim=1)  # Standard deviation
    else:
        raise ValueError("Invalid dev_type. Choose from 'unit_vec', 'sign', or 'std'.")
    
    gamma = torch.tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        if not distances:
            distances = distance[None, :]
        else:
            distances = torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)  # Maximum distance among benign updates.
    del distances

    # Finding optimal gamma using an iterative approach.
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = grad_mean - gamma * deviation
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2

    mal_update = grad_mean - gamma_succ * deviation

    for i in range(num_attackers_epoch):
        v[i] = mal_update[:, None]
    return v