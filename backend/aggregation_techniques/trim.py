import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging

from backend.aggregation_techniques.aggregation import update_global_model

logger = logging.getLogger(__name__)

def trim_mean(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Aggregates client gradients using the Trimmed Mean method.

    Based on the description in https://arxiv.org/abs/1803.01498, this function aggregates 
    client gradients by discarding the highest and lowest f gradients and computing the mean 
    of the remaining ones. The resulting global update is then used to update the global model.

    Args:
        gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): A list of tuples containing 
            each client identifier and its gradient dictionary.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Trim Mean aggregation but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are assumed malicious.
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.
    """
    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]
    n = len(gradients)
    logger.info(f"Aggregating gradients using Trim Mean with {f} malicious clients.")

    if n <= 2 * f:
        logger.error("Number of clients must be greater than 2f for Trim Mean aggregation.")
        raise ValueError("Insufficient number of clients for Trim Mean aggregation.")

    sorted_grads, _ = torch.sort(torch.cat(gradients, dim=1).to(device), dim=-1)
    global_update = torch.mean(sorted_grads[:, f:(n - f)], dim=-1)

    # Update the global model
    update_global_model(net, global_update, device)