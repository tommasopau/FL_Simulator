import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging




logger = logging.getLogger(__name__)


def update_global_model(net: nn.Module, global_update: torch.Tensor, device: torch.device) -> None:
    """
    Updates the global model parameters with the aggregated global update.

    Args:
        net (nn.Module): The global model to be updated.
        global_update (torch.Tensor): The aggregated updates for the model parameters.
        device (torch.device): The device to perform computations on.
    """
    with torch.no_grad():
        idx = 0
        for param in net.parameters():
            param_length = param.numel()
            param_update = global_update[idx:idx + param_length].reshape(param.size()).to(device)
            param.add_(param_update)
            idx += param_length
    logger.info("Global model parameters updated.")


def fedavg(
    gradients: List[torch.Tensor],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    data_sizes: List[int],
    **kwargs
) -> None:
    """
    Federated Averaging (FedAvg) aggregation method.

    Based on the description in https://arxiv.org/abs/1602.05629

    Args:
        gradients (List[torch.Tensor]): List of gradients from the clients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in FedAvg but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are malicious.
                   (Unused in FedAvg but kept for compatibility).
        device (torch.device): Computation device.
        data_sizes (List[int]): Number of training data samples per client.
        **kwargs: Additional keyword arguments.
    """
    if len(gradients) != len(data_sizes):
        logger.error("The number of gradients must match the number of data sizes.")
        raise ValueError("Mismatch between gradients and data_sizes lengths.")

    n = len(gradients)
    total_data_size = sum(data_sizes)
    with torch.no_grad():
          # compute global model update
          global_update = torch.zeros(gradients[0].size()).to(device)
          for i, grad in enumerate(gradients):
              global_update += grad * data_sizes[i]
          global_update /= total_data_size


    # Update the global model
    update_global_model(net, global_update, device)
    


def krum(
    gradients: List[torch.Tensor],
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
    


def median_aggregation(
    gradients: List[torch.Tensor],
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

    global_update, _ = torch.median(torch.cat(gradients, dim=1), dim=-1)
    
    # Update the global model
    update_global_model(net, global_update, device)
    

def trim_mean(
    gradients: List[torch.Tensor],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Trimmed Mean aggregation method.

    Based on the description in https://arxiv.org/abs/1803.01498

    Args:
        gradients (List[torch.Tensor]): List of gradients from the clients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate (unused in Trim Mean but kept for compatibility).
        f (int): Number of malicious clients. The first f clients are malicious.
        device (torch.device): Computation device.
        **kwargs: Additional keyword arguments.
    """
    n = len(gradients)
    logger.info(f"Aggregating gradients using Trim Mean with {f} malicious clients.")

    if n <= 2 * f:
        logger.error("Number of clients must be greater than 2f for Trim Mean aggregation.")
        raise ValueError("Insufficient number of clients for Trim Mean.")

    sorted, _ = torch.sort(torch.cat(gradients, dim=1).to(device), dim=-1)
    global_update = torch.mean(sorted[:, f:(n - f)], dim=-1)

    # Update the global model
    update_global_model(net, global_update, device)
    