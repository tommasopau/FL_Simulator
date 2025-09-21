import torch
from torch import nn
from typing import List, Dict,  Tuple
import logging

from fl_simulator.aggregation_techniques.aggregation import update_global_model

logger = logging.getLogger(__name__)


def median_aggregation(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> torch.Tensor:
    """
    Aggregates client gradients using the median method.

    Based on the description in https://arxiv.org/abs/1803.01498,
    this function aggregates gradients by computing the median across the concatenated client updates.
    The resulting global update is then used to update the global model.

    Args:
        gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): A list of tuples where each tuple contains
            a client identifier and its gradient dictionary.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Median aggregation but kept for compatibility).
        f (int): Number of malicious clients (unused in Median aggregation but kept for compatibility).
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The aggregated global update computed as the median of client gradients.
    """
    logger.info("Aggregating gradients using Median.")
    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]

    global_update, _ = torch.median(torch.cat(gradients, dim=1), dim=-1)
    update_global_model(net, global_update, device)
    return global_update
