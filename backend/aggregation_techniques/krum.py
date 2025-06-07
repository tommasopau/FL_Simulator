import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
from backend.aggregation_techniques.aggregation import update_global_model

    





logger = logging.getLogger(__name__)

def krum(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Krum aggregation method.

    Based on the description in https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html

    Args:
        gradients (List[torch.Tensor]): List of gradients from the clients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Krum but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are malicious.
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.
    """
    n = len(gradients)
    logger.info(f"Aggregating gradients using Krum considering {n} clients with {f} malicious.")

    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]

    # compute pairwise Euclidean distance
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(gradients[i] - gradients[j])
            dist[i, j], dist[j, i] = d, d

    # sort distances and get model with smallest sum of distances to closest n-f-2 models
    sorted_dist, _ = torch.sort(dist, dim=-1)
    global_update = gradients[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    
    # Update the global model
    update_global_model(net, global_update, device)