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
    Perform Krum aggregation on client gradients.

    This method selects a gradient that minimizes the sum of Euclidean distances 
    to its closest (n - f - 1) neighbors, where n is the number of clients and f is 
    the number of malicious clients. The chosen gradient is then used to update 
    the global model.

    Args:
        gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): List of tuples 
            containing each client identifier and its gradient dictionary.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Krum but kept for compatibility).
        f (int): Number of malicious clients.
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.
    """
    n = len(gradients)
    logger.info(f"Aggregating gradients using Krum considering {n} clients with {f} malicious.")

    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]

    # Compute pairwise Euclidean distances
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(gradients[i] - gradients[j])
            dist[i, j], dist[j, i] = d, d

    # Sort distances and select the gradient with the smallest sum of distances to its closest (n - f - 1) neighbors
    sorted_dist, _ = torch.sort(dist, dim=-1)
    global_update = gradients[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    
    # Update the global model
    update_global_model(net, global_update, device)