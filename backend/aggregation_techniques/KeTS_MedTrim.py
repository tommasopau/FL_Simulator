import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch.nn.functional as F
from numpy import array, linspace

import numpy as np

from scipy.signal import argrelextrema
from sklearn.cluster import estimate_bandwidth
from backend.utils.utility import segmentation


from backend.aggregation_techniques.median import median_aggregation
from backend.aggregation_techniques.trim import trim_mean

    





logger = logging.getLogger(__name__)

def KeTS_MedTrim(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    trust_scores: Dict[int,float],
    last_updates: Dict[int, torch.Tensor],
    baseline_decreased_score: float,
    **kwargs
) -> None:
    """
    KeTS aggregation method + Median/Trimmed Mean aggregation.

    Args:
        gradients (List[torch.Tensor]): List of gradients from the clients.
        net (nn.Module): The global model to be updated.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        device (torch.device): Computation device.
        trust_scores (List[float]): Trust scores for each client.
        last_updates (Dict[int, torch.Tensor]): Last updates from each client.
        **kwargs: Additional keyword arguments.
    """
    logger.info("Aggregating gradients using KeTS.")
    server = kwargs.get('last_global_update', None)
    additional_check = 'median'
    # Update trust scores for sampled clients
    if all(update is None for update in last_updates.values()):
        if additional_check == 'median':
            median_aggregation(gradients, net, lr, f, device, **kwargs)
        else:
            trim_mean(gradients, net, lr, f, device, **kwargs)
        return 
    for cid,gradient in gradients:
        if last_updates[cid] is not None:
            flat_update1 = gradient['flattened_diffs'].view(-1)
            flat_update2 = last_updates[cid].view(-1)
            
            # Compute cosine similarity
            sim = F.cosine_similarity(flat_update1, flat_update2, dim=0).item()
            # Compute Euclidean distance
            dist = torch.norm(flat_update1 - flat_update2).item()
            #logger.info(f"client {cid}, cosine similarity: {sim}, Euclidean distance: {dist}")
            if sim >= 0:
                alpha = (1 - sim) + dist
                trust_scores[cid] = max(0, trust_scores[cid] - baseline_decreased_score * alpha )
            else:
                trust_scores[cid] = 0
    logger.info(f"Updated trust scores: {trust_scores}")

            
    trust_scores_sampled = np.array([trust_scores[cid] for cid,_ in gradients])
    last_segment = segmentation(trust_scores_sampled , 'gaussian')
    
    honest_updates = [(cid,gradient) for cid,gradient in gradients if trust_scores[cid] >= last_segment]
    logger.info(f"Attacker clients: {[cid for cid,_ in gradients if trust_scores[cid] < last_segment]}")
    if additional_check == 'median':
            median_aggregation(honest_updates, net, lr, f, device, **kwargs)
    else:
            trim_mean(honest_updates, net, lr, f, device, **kwargs)
    return