import torch
from torch import nn
from typing import List, Dict, Tuple
import logging
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from backend.utils.utility import segmentation
from backend.aggregation_techniques.aggregation import update_global_model
from backend.aggregation_techniques.fedavg import fedavg

logger = logging.getLogger(__name__)

def detect_colluding_gradients(gradients: List[Tuple[int, Dict]]) -> Dict[int, int]:
    """
    Computes colluding scores for each client based on cosine similarity.
    """
    colluding_scores = {}
    n = len(gradients)
    for i in range(n):
        cid_i, grad_i = gradients[i]
        flat_i = grad_i['flattened_diffs'].view(-1)
        score = 0
        for j in range(n):
            if i == j:
                continue
            _, grad_j = gradients[j]
            flat_j = grad_j['flattened_diffs'].view(-1)
            sim = F.cosine_similarity(flat_i, flat_j, dim=0).item()
            if sim > 0.91:
                score += 1
        colluding_scores[cid_i] = score
    return colluding_scores

def update_trust_scores(
    trust_scores: Dict[int, float],
    gradients: List[Tuple[int, Dict]],
    last_updates: Dict[int, torch.Tensor],
    baseline_decreased_score: float
) -> Dict[int, float]:
    """
    Update each client's trust score based on current and last updates.
    """
    for cid, gradient in gradients:
        if last_updates[cid] is not None:
            flat_current = gradient['flattened_diffs'].view(-1)
            flat_last = last_updates[cid].view(-1)
            sim = F.cosine_similarity(flat_current, flat_last, dim=0).item()
            dist = torch.norm(flat_current - flat_last).item()
            if sim >= 0:
                alpha = (1 - sim) + dist
                trust_scores[cid] = max(0, trust_scores[cid] - baseline_decreased_score * alpha)
            else:
                trust_scores[cid] = 0
    return trust_scores

def filter_by_global_similarity(
    honest_updates: List[Tuple[int, Dict]],
    trust_scores2: Dict[int, float],
    server: object
) -> List[Tuple[int, Dict]]:
    """
    Filters honest_updates based on their cosine similarity with the last global update.
    """
    sim_with_global = defaultdict(float)
    if server is not None and getattr(server, 'last_global_update', None) is not None:
        global_flat = server.last_global_update.view(-1)
        for cid, grad in honest_updates:
            flat_update = grad['flattened_diffs'].view(-1)
            cos_sim = F.cosine_similarity(flat_update, global_flat, dim=0).item()
            sim_with_global[cid] = cos_sim
            
            euc_dist = torch.norm(flat_update - global_flat).item()
            logger.info(f"client {cid}, cosine similarity with global: {cos_sim:.4f}, Euclidean distance with global: {euc_dist:.4f}")
            '''
            abnormality = (1 - cos_sim) + euc_dist
            trust_scores2[cid] = max(0,trust_scores2[cid] - 0.1 * abnormality)
            '''
            trust_scores2[cid] = 0 if cos_sim < 0.1 else trust_scores2[cid]  # Reset if similarity is too low
            
    # Compute softmax weights from the updated trust_scores2.
    exp_scores = {cid: np.exp(score) for cid, score in trust_scores2.items()}
    total_exp = sum(exp_scores.values())
    weights = {cid: exp_scores[cid] / total_exp for cid in exp_scores}
        
    
    
    return weights

def KeTSV2(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    trust_scores: Dict[int, float],
    trust_scores2: Dict[int, float],
    last_updates: Dict[int, torch.Tensor],
    baseline_decreased_score: float,
    **kwargs
) -> None:
    logger.info("Aggregating gradients using KeTSV2.")
    server = kwargs.get('last_global_update', None)
    
    # Phase 1: First round using median aggregator when all last_updates are None.
    if all(update is None for update in last_updates.values()):
        colluding_scores = detect_colluding_gradients(gradients)
        logger.info(f"Colluding scores: {colluding_scores}")
        
        non_colluding_gradients = [(cid, grad) for cid, grad in gradients if colluding_scores[cid] == 0]
        if not non_colluding_gradients:
            logger.warning("All clients detected as colluding. Falling back to aggregating all gradients.")
            non_colluding_gradients = gradients
        
        grads = [grad['flattened_diffs'] for _, grad in non_colluding_gradients] #non_colluding_gradients]
        stacked = torch.stack(grads, dim=0)
        global_update, _ = torch.median(stacked, dim=0)
        update_global_model(net, global_update, device)
        server.last_global_update = global_update
        return
    
    # Phase 2: Subsequent rounds using FedAvg with trust score updates.
    # Update trust scores based on the similarity between current and last updates.
    trust_scores = update_trust_scores(trust_scores, gradients, last_updates, baseline_decreased_score)
    logger.info(f"Updated trust scores: {trust_scores}")

    # Determine segmentation threshold using trust scores.
    trust_scores_sampled = np.array([trust_scores[cid] for cid, _ in gradients])
    last_segment = segmentation(trust_scores_sampled, 'gaussian')
    logger.info(f"Segmentation threshold vertical analyses: {last_segment}")
    
    # Detect colluding clients.
    colluding_scores = detect_colluding_gradients(gradients)
    logger.info(f"Colluding scores: {colluding_scores}")
    
    non_colluding_gradients = [(cid, grad) for cid, grad in gradients if colluding_scores[cid] == 0]
    if not non_colluding_gradients:
        logger.warning("All clients detected as colluding. Falling back to aggregating all gradients.")
        non_colluding_gradients = gradients
    gradients = non_colluding_gradients
    
    # Filter clients considered "honest" based on trust scores.
    honest_updates = [(cid, grad) for cid, grad in gradients if trust_scores[cid] >= last_segment]
    logger.info(f"Attacker clients vertical analyses: {[cid for cid, _ in gradients if trust_scores[cid] < last_segment]}")
    
    # Further filter by cosine similarity with the last global update.
    weights = filter_by_global_similarity(honest_updates, trust_scores2,server)
    logger.info(f'TrustScores2 after filtering: {trust_scores2}')
    # Aggregate weighted gradients.
    agg_update = sum(weights[cid] * grad['flattened_diffs'] for cid, grad in honest_updates)
    global_update = agg_update
    update_global_model(net, global_update, device)
    logger.info(f"Aggregated clients {[cid for cid, _ in honest_updates]}")
    
    
    ema_alpha = 0.8
    if server is not None:
        if server.last_global_update is None:
            server.last_global_update = global_update
        else:
            server.last_global_update = ema_alpha * global_update + (1 - ema_alpha) * server.last_global_update