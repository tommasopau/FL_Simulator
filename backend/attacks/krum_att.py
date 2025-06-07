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
    Local model poisoning attack against the Krum aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        net (torch.nn.Module): Model.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
    """
    threshold = 1e-5

    n = len(v)
    w_re = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0)
    d = v[0].size()[0]
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):  # compute euclidean distance of benign to benign devices
        for j in range(i + 1, n):
            d = torch.norm(lr * v[i] - lr * v[j], p=2)
            dist[i, j], dist[j, i] = d, d

    dist_benign_sorted, _ = torch.sort(dist[f:, f:])
    min_dist = torch.min(torch.sum(dist_benign_sorted[:, 0:(n - f - 1)], dim=-1))

    dist_w_re = []
    for i in range(f, n):
        dist_w_re.append(torch.norm(lr * v[i], p=2))
    max_dist_w_re = torch.max(torch.stack(dist_w_re))

    max_lambda = min_dist / ((n - 2 * f - 1) * torch.sqrt(d)) + max_dist_w_re / torch.sqrt(d)

    actual_lambda = max_lambda
    sorted_dist, _ = torch.sort(dist, dim=-1)
    update_before = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    while actual_lambda > threshold:
        for i in range(num_attackers_epoch):
            v[i] = - actual_lambda * torch.sign(update_before)

        dist = torch.zeros((n, n)).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(v[i] - v[j])
                dist[i, j], dist[j, i] = d, d
        sorted_dist, _ = torch.sort(dist, dim=-1)
        global_update = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
        if torch.equal(global_update, v[0]):
            break
        else:
            actual_lambda = actual_lambda / 2

    return v