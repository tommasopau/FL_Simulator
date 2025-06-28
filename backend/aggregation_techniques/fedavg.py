import torch
from torch import nn
from typing import List, Dict, Tuple
import logging


from backend.aggregation_techniques.aggregation import update_global_model

logger = logging.getLogger(__name__)


def fedavg(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Federated Averaging (FedAvg) aggregation method.

    Based on the description in https://arxiv.org/abs/1602.05629
    
    Args:
        gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): 
            List of tuples containing client ID and their gradients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in FedAvg but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are malicious.
                 (Unused in FedAvg but kept for compatibility).
        device (torch.device): The device on which computations are performed.
        **kwargs: Additional keyword arguments.

    
    """
    logger.info("Aggregating gradients using FedAvg.")
    
    total_data_size = sum([gradient[1]['data_size'] for gradient in gradients])
    with torch.no_grad():
          # compute global model update
          global_update = torch.zeros(gradients[0][1]['flattened_diffs'].size()).to(device)
          for gradient in gradients:
              global_update += gradient[1]['flattened_diffs'] * gradient[1]['data_size']
          global_update /= total_data_size


    # Update the global model
    update_global_model(net, global_update, device)
    return global_update
    