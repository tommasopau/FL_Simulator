import torch 
from typing import List


def min_sum_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning).
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
    """
    
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    if dev_type == 'unit_vec':
        deviation = grad_mean / torch.norm(grad_mean, p=2)  # unit vector, dir opposite to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(grad_mean)  # signs of the gradients
    elif dev_type == 'std':
        deviation = torch.std(catv, dim=1)  # standard deviation
    else:
        raise ValueError("Invalid dev_type. Choose from 'unit_vec', 'sign', or 'std'.")


    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)

    for i in range(num_attackers_epoch):
        v[i] = mal_update[:, None]
    return v