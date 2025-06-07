import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch.nn.functional as F

import numpy as np

from backend.utils.utility import segmentation
from collections import defaultdict



    





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
    
    n = len(gradients)
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
    

def trim_mean(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
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
    
    gradients = [gradient[1]['flattened_diffs'] for gradient in gradients]
    n = len(gradients)
    logger.info(f"Aggregating gradients using Trim Mean with {f} malicious clients.")

    if n <= 2 * f:
        logger.error("Number of clients must be greater than 2f for Trim Mean aggregation.")
        raise ValueError("Insufficient number of clients for Trim Mean.")

    sorted, _ = torch.sort(torch.cat(gradients, dim=1).to(device), dim=-1)
    global_update = torch.mean(sorted[:, f:(n - f)], dim=-1)

    # Update the global model
    update_global_model(net, global_update, device)


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
    



def KeTSV2(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    trust_scores: Dict[int, float],
    last_updates: Dict[int, torch.Tensor],
    baseline_decreased_score: float,
    **kwargs
) -> None:
    logger.info("Aggregating gradients using KeTSV2.")
    server = kwargs.get('last_global_update', None)
    
    

    
    '''
    check_clients = [30] #3132
    if check_clients is not None and all(update is None for update in last_updates.values()):
        honest_updates = gradients
        logger.info(f"Performing additional check step using check list: {check_clients}")
        
        # Identify checked clients from the honest updates.
        checked = {cid: grad for cid, grad in honest_updates if cid in check_clients}
        # For each client not in the check list, compute the sum of cosine similarities with each checked client.
        sum_cosine = {}
        for cid, grad in honest_updates:
            if cid in checked:
                continue  # Do not compare two check clients.
            flat_update = grad['flattened_diffs'].view(-1)
            sim_sum = 0.0
            for chk_cid, chk_grad in checked.items():
                flat_chk = chk_grad['flattened_diffs'].view(-1)
                sim_val = F.cosine_similarity(flat_update, flat_chk, dim=0).item()
                sim_sum += sim_val
            sum_cosine[cid] = sim_sum
        
        # Normalize sum_cosine values to be between 0 and 1
        max_sum = max(sum_cosine.values())
        min_sum = min(sum_cosine.values())
        normalized_sum_cosine = {cid: (total - min_sum) / (max_sum - min_sum) for cid, total in sum_cosine.items()}
        
        # Sort non-checked clients by the normalized sum (in descending order).
        sorted_sum = sorted(normalized_sum_cosine.items(), key=lambda x: x[1], reverse=True)
        
        ls = segmentation(np.array([total  for _, total in sorted_sum]),'exponential')
        logger.info(f"Segmentation threshold: {ls}")
        logger.info("Sorted sum of cosine similarities:")
        for cid, total in sorted_sum:
            logger.info(f"Client {cid}: {total:.6f}")
        filtered = []
        for cid, total in sorted_sum:
            if total >= ls:
                filtered.append(cid)
        
        logger.info(f"Filtered clients: {filtered}")
        gradients = [(cid, grad) for cid, grad in honest_updates if cid in filtered]
        global_update = fedavg(gradients, net, lr, f, device, **kwargs)
        server.last_global_update = global_update
        return
    '''
    #MEDIAN round 1
    if all(update is None for update in last_updates.values()):
        #-- Still checking colluding in the first round
        colluding_scores = {}
        n = len(gradients)
        for i in range(n):
            cid_i, grad_i = gradients[i]
            score = 0
            flat_i = grad_i['flattened_diffs'].view(-1)
            for j in range(n):
                if i == j:
                    continue
                _, grad_j = gradients[j]
                flat_j = grad_j['flattened_diffs'].view(-1)
                sim = F.cosine_similarity(flat_i, flat_j, dim=0).item()
                if sim > 0.99:
                    score += 1
            colluding_scores[cid_i] = score
        logger.info(f"Colluding scores: {colluding_scores}")
        
        # --- Remove colluding clients from aggregation ---
        non_colluding_gradients = [(cid, grad) for cid, grad in gradients if colluding_scores[cid] == 0]
        if not non_colluding_gradients:
            logger.warning("All clients detected as colluding. Falling back to aggregating all gradients.")
            non_colluding_gradients = gradients
        gradients = non_colluding_gradients
        
        
        grads = [gradient[1]['flattened_diffs'] for gradient in gradients]
        # Stack gradients into a tensor of shape (n_clients, n_params)
        stacked = torch.stack(grads, dim=0)
        global_update, _ = torch.median(stacked, dim=0)
        update_global_model(net, global_update, device)
        server.last_global_update = global_update
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
    logger.info(f"Segmentation threshold vertycal analyses: {last_segment}")
    #--Colluding detection
    
    # --- Compute colluding metrics ---
    # For each client, compute cosine similarity with all other clients.
    # For each similarity > 0.90, increase its colluding score.
    colluding_scores = {}
    n = len(gradients)
    for i in range(n):
        cid_i, grad_i = gradients[i]
        score = 0
        flat_i = grad_i['flattened_diffs'].view(-1)
        for j in range(n):
            if i == j:
                continue
            _, grad_j = gradients[j]
            flat_j = grad_j['flattened_diffs'].view(-1)
            sim = F.cosine_similarity(flat_i, flat_j, dim=0).item()
            if sim > 0.99:
                score += 1
        colluding_scores[cid_i] = score
    logger.info(f"Colluding scores: {colluding_scores}")
    
    # --- Remove colluding clients from aggregation ---
    non_colluding_gradients = [(cid, grad) for cid, grad in gradients if colluding_scores[cid] == 0]
    if not non_colluding_gradients:
        logger.warning("All clients detected as colluding. Falling back to aggregating all gradients.")
        non_colluding_gradients = gradients
    gradients = non_colluding_gradients
    
    
    honest_updates = [(cid,gradient) for cid,gradient in gradients if trust_scores[cid] >= last_segment]
    logger.info(f"Attacker clients vertical analyses: {[cid for cid,_ in gradients if trust_scores[cid] < last_segment]}")
    sim_with_global = defaultdict(float)
    if server is not None and getattr(server, 'last_global_update', None) is not None:
        final_updates = []
        for cid, grad in honest_updates:
            flat_update = grad['flattened_diffs'].view(-1)
            global_flat = server.last_global_update.view(-1)
            cos_sim_global = F.cosine_similarity(flat_update, global_flat, dim=0).item()
            sim_with_global[cid] = cos_sim_global
            
            euc_dist_global = torch.norm(flat_update - global_flat).item()
            logger.info(f"client {cid}, cosine similarity with global: {cos_sim_global}, Euclidean distance with global: {euc_dist_global}")
    if sim_with_global:
        #server_segment = segmentation(np.array(list(sim_with_global.values())) , 'exponential')
        honest_updates = [(cid,gradient) for cid,gradient in honest_updates if sim_with_global[cid] >= 0.1]
    logger.info(f"Aggregated clients {[cid for cid,_ in honest_updates]}")
    
    global_update = fedavg(honest_updates, net, lr, f, device, **kwargs)
    ema_alpha = 0.2
    if server is not None:
        if server.last_global_update is None:
            server.last_global_update = global_update
        else:
            server.last_global_update = ema_alpha * global_update + (1 - ema_alpha) * server.last_global_update

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