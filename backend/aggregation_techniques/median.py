import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging


from backend.aggregation_techniques.aggregation import update_global_model

    





logger = logging.getLogger(__name__)

def median_aggregation(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Median aggregation method.

    Based on the description in https://arxiv.org/abs/1803.01498

    Args:
        gradients (List[torch.Tensor]): List of gradients from the clients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Median aggregation but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are malicious.
                   (Unused in Median aggregation but kept for compatibility).
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.
    """
    logger.info("Aggregating gradients using Median.")
    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]

    global_update, _ = torch.median(torch.cat(gradients, dim=1), dim=-1)
    
    
    update_global_model(net, global_update, device)
    
    return global_update