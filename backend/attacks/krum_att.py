import torch 
import numpy as np
from typing import List


def krum_attack(
    v: List[torch.Tensor],
    net: torch.nn.Module,
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Performs a local model poisoning attack against the Krum aggregation rule.

    This attack modifies the gradients of malicious clients by first computing
    the pairwise Euclidean distances among gradients, estimating a scaling factor,
    and then updating the malicious gradients by applying a negative scaled sign
    of a selected benign update. The process is repeated until a threshold is reached.
    
    Based on the description in https://arxiv.org/abs/1911.11815.

    Args:
        v (List[torch.Tensor]): List of client gradients.
        net (torch.nn.Module): The model from which gradient dimensions are derived.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f gradients in v are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used for training and inference.

    Returns:
        List[torch.Tensor]: The modified list of gradients with the attack applied.
    """
    threshold = 1e-5
    n = len(v)
    w_re = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0)
    d = v[0].size()[0]
    dist = torch.zeros((n, n)).to(device)
    
    # Compute Euclidean distances between gradients.
    for i in range(n):
        for j in range(i + 1, n):
            d_val = torch.norm(lr * v[i] - lr * v[j], p=2)
            dist[i, j], dist[j, i] = d_val, d_val

    dist_benign_sorted, _ = torch.sort(dist[f:, f:])
    min_dist = torch.min(torch.sum(dist_benign_sorted[:, 0:(n - f - 1)], dim=-1))

    dist_w_re = []
    for i in range(f, n):
        dist_w_re.append(torch.norm(lr * v[i], p=2))
    max_dist_w_re = torch.max(torch.stack(dist_w_re))
    
    # Ensure using float for sqrt computations.
    sqrt_d = torch.sqrt(torch.tensor(d, dtype=torch.float32))
    max_lambda = min_dist / ((n - 2 * f - 1) * sqrt_d) + max_dist_w_re / sqrt_d
    actual_lambda = max_lambda

    sorted_dist, _ = torch.sort(dist, dim=-1)
    update_before = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    
    while actual_lambda > threshold:
        for i in range(num_attackers_epoch):
            v[i] = - actual_lambda * torch.sign(update_before)

        dist = torch.zeros((n, n)).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                d_val = torch.norm(v[i] - v[j])
                dist[i, j], dist[j, i] = d_val, d_val

        sorted_dist, _ = torch.sort(dist, dim=-1)
        global_update = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
        if torch.equal(global_update, v[0]):
            break
        else:
            actual_lambda = actual_lambda / 2

    return v