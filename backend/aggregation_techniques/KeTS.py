import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch.nn.functional as F

import numpy as np

from backend.utils.utility import segmentation

from backend.aggregation_techniques.aggregation import update_global_model
from backend.aggregation_techniques.fedavg import fedavg


    





logger = logging.getLogger(__name__)

def KeTS(
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
    KeTS aggregation method.

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
    # Update trust scores for sampled clients
    if all(update is None for update in last_updates.values()):
        fedavg(gradients, net, lr, f, device, **kwargs)
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
    fedavg(honest_updates, net, lr, f, device, **kwargs)
    

