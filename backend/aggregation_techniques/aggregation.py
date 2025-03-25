import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch.nn.functional as F
from numpy import array, linspace
from sklearn.neighbors import KernelDensity
import numpy as np
import scipy.signal as signal
from scipy.signal import argrelextrema
from sklearn.cluster import estimate_bandwidth
from utils.utility import segmentation
import scipy.fft
import pywt
from collections import defaultdict
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


    





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
            logger.info(f"client {cid}, cosine similarity: {sim}, Euclidean distance: {dist}")
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
    

def DCT(
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
    Discrete Cosine Transform (DCT) aggregation method.
    """
    modality = 3
    '''
    modality = 0: KeTS with DCT
    modality = 1: analyze single dct update
    modality = 2: analyze last global update
    '''
    server = kwargs.get('last_global_update', None)
    logger.info("Aggregating gradients using DCT.")
    features_list = []  # list to collect per-client features and A1 tensor
    # Update trust scores for sampled clients
    '''
    if all(update is None for update in last_updates.values()):
        fedavg(gradients, net, lr, f, device, **kwargs)
        return 
    '''
    a1_tensors = []
    if all(update is None for update in last_updates.values()):
        logger.info("No last updates available; performing component-wise median aggregation to avoid outliers.")
        grads = [gradient[1]['flattened_diffs'] for gradient in gradients]
        # Stack gradients into a tensor of shape (n_clients, n_params)
        stacked = torch.stack(grads, dim=0)
        global_update, _ = torch.median(stacked, dim=0)
        update_global_model(net, global_update, device)
        server.last_global_update = global_update
        return 
    else:
        for cid, gradient in gradients:
            
            if modality == 3 and last_updates[cid] is not None:
                flat_update1 = gradient['flattened_diffs'].view(-1)
                flat_update2 = last_updates[cid].view(-1)
                # Detach and move to CPU before converting to NumPy
                flat_update1_np = flat_update1.detach().cpu().numpy()
                flat_update2_np = flat_update2.detach().cpu().numpy()
                # Increase DWT levels from 2 to 4 by adding two more levels in the decomposition
                A1, D1, D2, D3, D4 = pywt.wavedec(flat_update1_np, 'haar', level=4)

                A2, D2_1, D2_2, D2_3, D2_4 = pywt.wavedec(flat_update2_np, 'haar', level=4)
                
                A1_tensor = torch.from_numpy(A1).to(device)
                A2_tensor = torch.from_numpy(A2).to(device)
                D1_tensor = torch.from_numpy(D1).to(device)
                D2_tensor = torch.from_numpy(D2).to(device)
                D3_tensor = torch.from_numpy(D3).to(device)
                D4_tensor = torch.from_numpy(D4).to(device)
                D2_1_tensor = torch.from_numpy(D2_1).to(device)
                D2_2_tensor = torch.from_numpy(D2_2).to(device)
                D2_3_tensor = torch.from_numpy(D2_3).to(device)
                D2_4_tensor = torch.from_numpy(D2_4).to(device)
                
                cosA = F.cosine_similarity(A1_tensor, A2_tensor, dim=0).item()
                cosD1 = F.cosine_similarity(D1_tensor, D2_1_tensor, dim=0).item()
                cosD2 = F.cosine_similarity(D2_tensor, D2_2_tensor, dim=0).item()
                cosD3 = F.cosine_similarity(D3_tensor, D2_3_tensor, dim=0).item()
                cosD4 = F.cosine_similarity(D4_tensor, D2_4_tensor, dim=0).item()
                
                euclidean_A = torch.norm(A1_tensor - A2_tensor).item()
                euclidean_D1 = torch.norm(D1_tensor - D2_1_tensor).item()
                euclidean_D2 = torch.norm(D2_tensor - D2_2_tensor).item()
                euclidean_D3 = torch.norm(D3_tensor - D2_3_tensor).item()
                euclidean_D4 = torch.norm(D4_tensor - D2_4_tensor).item()
                
                '''  
                logger.info(f"client {cid}, cosine similarity (A): {cosA:.3f}, (D1): {cosD1:.3f}, (D2): {cosD2:.3f}, (D3): {cosD3:.3f}, (D4): {cosD4:.3f}, "
                            f"Euclidean distance (A): {euclidean_A:.3f}, (D1): {euclidean_D1:.3f}, (D2): {euclidean_D2:.3f}, (D3): {euclidean_D3:.3f}, (D4): {euclidean_D4:.3f}")
                '''
                
                features_list.append({
                    "cid": cid,
                    "features": [cosA, cosD1, cosD2,cosD3,cosD4, euclidean_A, euclidean_D1, euclidean_D2 , euclidean_D3, euclidean_D4],
                    "A1": A1_tensor ,
                    "D1": D1_tensor,
                    "D2": D2_tensor,
                })
                if server.last_global_update is not None:
                    flat_global_update = server.last_global_update.view(-1)
                    flat_global_update_np = flat_global_update.detach().cpu().numpy()
                    A_global, D1_global , D2_global , D3_global , D4_global = pywt.wavedec(flat_global_update_np, 'haar', level=4)
                    server_cosA = F.cosine_similarity(A1_tensor, torch.from_numpy(A_global).to(device), dim=0).item()
                    server_cosD1 = F.cosine_similarity(D1_tensor, torch.from_numpy(D1_global).to(device), dim=0).item()
                    server_cosD2 = F.cosine_similarity(D2_tensor, torch.from_numpy(D2_global).to(device), dim=0).item()
                    server_cosD3 = F.cosine_similarity(D3_tensor, torch.from_numpy(D3_global).to(device), dim=0).item()
                    server_cosD4 = F.cosine_similarity(D4_tensor, torch.from_numpy(D4_global).to(device), dim=0).item()
                    
                    euclidean_A_global = torch.norm(A1_tensor - torch.from_numpy(A_global).to(device)).item()
                    euclidean_D1_global = torch.norm(D1_tensor - torch.from_numpy(D1_global).to(device)).item()
                    euclidean_D2_global = torch.norm(D2_tensor - torch.from_numpy(D2_global).to(device)).item()
                    euclidean_D3_global = torch.norm(D3_tensor - torch.from_numpy(D3_global).to(device)).item()
                    euclidean_D4_global = torch.norm(D4_tensor - torch.from_numpy(D4_global).to(device)).item()
                    
                    #logger.info(f"client {cid}, cosine similarity (A): {server_cosA:.3f}, (D1): {server_cosD1:.3f} , Euclidean distance (A): {euclidean_A_global:.3f}, (D1): {euclidean_D1_global:.3f}")
                    features_list[-1]["features"].extend([
                        server_cosA, server_cosD1, server_cosD2, server_cosD3, server_cosD4,
                        euclidean_A_global, euclidean_D1_global, euclidean_D2_global, euclidean_D3_global, euclidean_D4_global
                    ])
                    '''
                    # --- Added raw update comparisons ---
                    raw_update_current = gradient['flattened_diffs'].view(-1)
                    raw_update_last = server.last_global_update.view(-1)
                    raw_cos = F.cosine_similarity(raw_update_current, raw_update_last, dim=0).item()
                    raw_euc = torch.norm(raw_update_current - raw_update_last).item()
                    logger.info(f"client {cid}, raw cosine similarity: {raw_cos:.3f}, raw Euclidean distance: {raw_euc:.3f}")
                    #features_list[-1]["features"].extend([raw_cos, raw_euc])
                    '''
        
        
    
            
            
    

    num_clients = len(features_list)
    colluding_scores = defaultdict(int)  # dict to store colluding counts per client in modality 3
    if modality == 3:
        for i in range(num_clients):
            cos_sim_tot_A1 = 0.0
            euc_dist_tot_A1 = 0.0
            collud_count = 0  # count of times cosine similarity > 0.8
            for j in range(num_clients):
                if i == j:
                    continue
                # A1 calculations
                a1_i = features_list[i]["A1"]
                a1_j = features_list[j]["A1"]
                cos_sim_A1 = F.cosine_similarity(a1_i, a1_j, dim=0).item()
                euc_A1 = torch.norm(a1_i - a1_j).item()
                cos_sim_tot_A1 += (1 - cos_sim_A1)
                euc_dist_tot_A1 += euc_A1
                # Count colluding occurrences if cosine similarity > 0.8
                if cos_sim_A1 > 0.9:
                    collud_count += 1

                
            # Append total metrics for A1, D1 and D2 in order
            '''
            features_list[i]["features"].extend([
                cos_sim_tot_A1
            ]) #euc_dist_tot_A1
            '''
            
            colluding_scores[features_list[i]['cid']] = collud_count

        logger.info(f"Colluding scores: {colluding_scores}")
    
    

    # Perform clustering using HDBSCAN
    cids = []
    features = []
    for item in features_list:
        cids.append(item['cid'])
        features.append(item['features'])

    if features_list:
        # Convert list to numpy array and normalize
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=3 , min_samples=4)
        cluster_labels = clusterer.fit_predict(X_scaled)

        # Map each cluster label to its corresponding client IDs
        clusters = {}
        for cid, label in zip(cids, cluster_labels):
            clusters.setdefault(label, []).append(cid)
    
        # Log the cluster information
        logger.info(f"Clusters formed: {clusters}")
        
        # Define feature names corresponding to each index in the feature vector.
        # For modality 3, if you are appending features in this order:
        # [cosA, cosD1, cosD2, cosD3, cosD4,
        #  euclidean_A, euclidean_D1, euclidean_D2, euclidean_D3, euclidean_D4,
        #  server_cosA, server_cosD1, server_cosD2, server_cosD3, server_cosD4,
        #  euclidean_A_global, euclidean_D1_global, euclidean_D2_global, euclidean_D3_global, euclidean_D4_global,
        #  cos_sim_tot_A1, euc_dist_tot_A1]
        feature_names = [
            "cosA", "cosD1", "cosD2", "cosD3", "cosD4",
            "euclidean_A", "euclidean_D1", "euclidean_D2", "euclidean_D3", "euclidean_D4",
            "server_cosA", "server_cosD1", "server_cosD2", "server_cosD3", "server_cosD4",
            "euclidean_A_global", "euclidean_D1_global", "euclidean_D2_global", "euclidean_D3_global", "euclidean_D4_global",
            "cos_sim_tot_A1", "euc_dist_tot_A1"
        ]
        
        # Group features by cluster and log each mean along with the feature name
        cluster_features = {}
        for label, feat in zip(cluster_labels, features):
            cluster_features.setdefault(label, []).append(feat)
        for label, feat_list in cluster_features.items():
            mean_feat = np.mean(feat_list, axis=0)
            # Build a string that pairs each mean value with its feature name.
            mean_str = ", ".join(
                [f"{name}: {value:.6f}"for name, value in zip(feature_names, mean_feat)]
            )
            logger.info(f"Cluster {label} mean features: {mean_str}")
            
        cluster_scores = {}
        for label, feat_list in cluster_features.items():
            if label == -1:
                continue
            mean_feat = np.mean(feat_list, axis=0)
            # Feature indices:
            # 0-4: cosA, cosD1, cosD2, cosD3, cosD4
            # 5-9: euclidean_A, euclidean_D1, euclidean_D2, euclidean_D3, euclidean_D4
            # 10-14: server_cosA, server_cosD1, server_cosD2, server_cosD3, server_cosD4
            # 15-19: euclidean_A_global, euclidean_D1_global, euclidean_D2_global, euclidean_D3_global, euclidean_D4_global
            # 20: cos_sim_tot_A1
            # 21: euc_dist_tot_A1
            score = (
                (mean_feat[0] ) +
                (mean_feat[10] + mean_feat[11]) -
                (mean_feat[5]) -
                (mean_feat[15] + mean_feat[16])
            )
            cluster_scores[label] = score
            logger.info(f"Cluster {label} score: {score:.6f}")

        # Determine the best cluster by taking the one with maximum score
        best_cluster = max(cluster_scores, key=cluster_scores.get)
        logger.info(f"Best cluster selected for aggregation: {best_cluster}")
    
        # Compute the mean flattened update for each cluster.
        grad_dict = {cid: grad for cid, grad in gradients}
        cluster_means = {}
        for label, cid_list in clusters.items():
            # Skip noise cluster (-1) if desired
            if label == -1:
                continue
            updates = [grad_dict[cid]['flattened_diffs'] for cid in cid_list]
            updates_stack = torch.stack(updates, dim=0)
            cluster_means[label] = torch.mean(updates_stack, dim=0)
            
            

        # Compute pairwise cosine similarity between cluster means if there are multiple clusters
        if len(cluster_means) > 1:
            cluster_keys = list(cluster_means.keys())
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    label_i = cluster_keys[i]
                    label_j = cluster_keys[j]
                    cos_sim = F.cosine_similarity(
                        cluster_means[label_i].view(1, -1),
                        cluster_means[label_j].view(1, -1)
                    ).item()
                    logger.info(f"Cosine similarity between cluster {label_i} and cluster {label_j}: {cos_sim:.6f}")
        
            
            
    
    if features_list:
        # After selecting the best cluster, filter the gradients based on the best cluster's cids
        selected_cids = clusters.get(best_cluster, [])
        logger.info(f"Selected client IDs for aggregation: {selected_cids}")

        filtered_gradients = [gradient for gradient in gradients if gradient[0] in selected_cids]
    else:
        filtered_gradients = gradients
        
    global_update = fedavg(filtered_gradients, net, lr, f, device, **kwargs)

    ema_alpha = 0.3
    if server.last_global_update is None:
        server.last_global_update = global_update
    else:
        server.last_global_update = ema_alpha * global_update + (1 - ema_alpha) * server.last_global_update


def DCT_K(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    trust_scores: Dict[int, float],
    last_updates: Dict[int, torch.Tensor],
    baseline_decreased_score: float,
    baseline_cosine: float = 0.85,
    **kwargs
) -> None:
    """
    DCT aggregation variant using KMeans (with k = 2). After clustering, the mean update (flattened_diffs)
    for each cluster is computed and their cosine similarity is measured. If the similarity is above a given
    baseline, the clusters are merged (i.e. aggregated together).
    """
    # Inherit the server from kwargs to support EMA global update update (if available)
    server = kwargs.get('server', None)
    logger.info("Aggregating gradients using DCT (KMeans variant).")
    
    features_list = []  # list to collect per-client features (for clustering)
    
    # Build features_list from available gradient updates
    for cid, gradient in gradients:
        if last_updates.get(cid) is not None:
            flat_update1 = gradient['flattened_diffs'].view(-1)
            flat_update2 = last_updates[cid].view(-1)
            flat_update1_np = flat_update1.detach().cpu().numpy()
            flat_update2_np = flat_update2.detach().cpu().numpy()
            # Increase DWT levels to 4 as in modality 3
            A1, D1, D2, D3, D4 = pywt.wavedec(flat_update1_np, 'haar', level=4)
            A2, D2_1, D2_2, D2_3, D2_4 = pywt.wavedec(flat_update2_np, 'haar', level=4)
            
            A1_tensor = torch.from_numpy(A1).to(device)
            D1_tensor = torch.from_numpy(D1).to(device)
            D2_tensor = torch.from_numpy(D2).to(device)
            # Compute cosine similarities between current and last updates
            cosA = F.cosine_similarity(A1_tensor, torch.from_numpy(A2).to(device), dim=0).item()
            euclidean_A = torch.norm(A1_tensor - torch.from_numpy(A2).to(device)).item()
            
            # For simplicity, we use a feature vector with the cosine similarity and Euclidean distance.
            # (You can extend the feature vector as needed.)
            features = [cosA, euclidean_A]
            
            features_list.append({
                "cid": cid,
                "features": features,
            })
    
    if features_list:
        # Build feature matrix
        X = np.array([item["features"] for item in features_list])
        # Normalize features (recommended for KMeans)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform KMeans clustering with k = 2
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaled)
        cluster_labels = kmeans.labels_
                        
        # Map cluster label to list of client IDs
        clusters = {}
        for item, label in zip(features_list, cluster_labels):
            clusters.setdefault(label, []).append(item['cid'])
        
        logger.info(f"KMeans clusters formed: {clusters}")
        
        # Create a lookup for original gradients by client id.
        grad_dict = {cid: grad for cid, grad in gradients}
        
        # Compute the mean flattened update for each cluster.
        cluster_means = {}
        for label, cid_list in clusters.items():
            updates = [grad_dict[cid]['flattened_diffs'] for cid in cid_list]
            updates_stack = torch.stack(updates, dim=0)
            cluster_means[label] = torch.mean(updates_stack, dim=0)
        
        # Compute cosine similarity between the two cluster means
        if len(cluster_means) == 2:
            mean0 = cluster_means[0]
            mean1 = cluster_means[1]
            cos_sim = F.cosine_similarity(mean0.view(1, -1), mean1.view(1, -1)).item()
            logger.info(f"Cosine similarity between cluster means: {cos_sim:.6f}")
            
            if (cos_sim > baseline_cosine):
                logger.info("Clusters are similar (cosine sim > baseline), merging clusters for aggregation.")
                selected_cids = [item["cid"] for item in features_list]
            else:
                # Choose the cluster with more clients
                best_label = 0 if len(clusters[0]) >= len(clusters[1]) else 1
                logger.info(f"Selected cluster {best_label} for aggregation (not merging clusters).")
                selected_cids = clusters.get(best_label, [])
        else:
            # If KMeans did not form two clusters, fall back to all clients
            logger.info("Only one cluster formed; using all gradients.")
            selected_cids = [item["cid"] for item in features_list]
            
        # Filter gradients based on selected client IDs from clustering
        filtered_gradients = [grad for grad in gradients if grad[0] in selected_cids]
    else:
        filtered_gradients = gradients

    # Aggregate the updates using FedAvg
    global_update = fedavg(filtered_gradients, net, lr, f, device, **kwargs)
    
    # Exponential moving average of global updates
    ema_alpha = 0.3
    if server is not None:
        if server.last_global_update is None:
            server.last_global_update = global_update
        else:
            server.last_global_update = ema_alpha * global_update + (1 - ema_alpha) * server.last_global_update
def extract_last_layer_params(net: nn.Module, flattened_grad: torch.Tensor) -> torch.Tensor:
    # Get the last layer's weight and bias from the model parameters
    params = list(net.parameters())
    last_layer_weight = params[-2]
    last_layer_bias = params[-1]
    
    weight_size = last_layer_weight.numel()
    bias_size = last_layer_bias.numel()
    
    # Compute offset up to the last layer (both weight and bias)
    offset = sum(p.numel() for p in params[:-2])
    
    # Reshape the corresponding segments of the flattened tensor
    weight_update = flattened_grad[offset: offset + weight_size].view(-1)
    bias_update = flattened_grad[offset + weight_size: offset + weight_size + bias_size].view(-1)
    
    # Concatenate the flattened versions of weight_update and bias_update
    concatenated = torch.cat([weight_update.view(-1), bias_update.view(-1)])
    
    return concatenated


def DCT_raw(
    gradients: list,
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    trust_scores: dict,
    last_updates: dict,
    baseline_decreased_score: float,
    **kwargs
) -> None:
    """
    A variant of the DCT aggregation method that uses raw flattened updates instead of 
    applying a DWT transformation. All other operations (clustering, computing cosine 
    similarity, Euclidean distances and EMA update) remain as in the original method.
    
    Args:
        gradients (List[Tuple[int, Dict[str, torch.Tensor]]]): List of tuples containing client ID and their update dict.
        net (nn.Module): The global model.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        device (torch.device): Computation device.
        trust_scores (dict): Client trust scores.
        last_updates (dict): Last update tensors for each client.
        baseline_decreased_score (float): Parameter for decreasing trust scores.
        **kwargs: Additional keyword arguments (e.g. last_global_update).
    """
    # Using modality 3 flag for compatibility.
    modality = 3
    server = kwargs.get('last_global_update', None)
    logger.info("Aggregating gradients using DCT_raw (raw update comparisons).")
    features_list = []
    
    # In this implementation, we compare raw flattened updates without applying wavelet transforms.
    if not all(last_updates[cid] is None for cid, _ in gradients):
        for cid, gradient in gradients:
            if modality == 3 and last_updates[cid] is not None:
                flat_update = gradient['flattened_diffs'].view(-1)
                flat_last = last_updates[cid].view(-1)
                
                
                
                # Compute cosine similarity and Euclidean distance between raw updates
                cosA = F.cosine_similarity(flat_update, flat_last, dim=0).item()
                euclidean_A = torch.norm(flat_update - flat_last).item()
                
                features_list.append({
                    "cid": cid,
                    "update" : flat_update,
                    "features": [cosA, euclidean_A],
                })
                
                if server is not None:
                    flat_global_update = server.last_global_update.view(-1)
                    # Compare current update (raw) to last global update (raw)
                    server_cos = F.cosine_similarity(flat_update, flat_global_update, dim=0).item()
                    server_euclidean = torch.norm(flat_update - flat_global_update).item()
                    
                    #logger.info(f"client {cid}, raw cosine similarity to global update: {server_cosA:.3f}, "
                                #f"raw Euclidean distance: {euclidean_A_global:.3f}")
                    features_list[-1]["features"].extend([server_cos, server_euclidean])
                    
                    
    num_clients = len(features_list)
    colluding_scores = defaultdict(int)  # dict to store colluding counts per client in modality 3
    if modality == 3:
        for i in range(num_clients):
            cos_sim_tot_A1 = 0.0
            euc_dist_tot_A1 = 0.0
            collud_count = 0  # count of times cosine similarity > 0.8
            for j in range(num_clients):
                if i == j:
                    continue
                # A1 calculations
                a1_i = features_list[i]["update"]
                a1_j = features_list[j]["update"]
                cos_sim_A1 = F.cosine_similarity(a1_i, a1_j, dim=0).item()
                euc_A1 = torch.norm(a1_i - a1_j).item()
                cos_sim_tot_A1 += (1 - cos_sim_A1)
                euc_dist_tot_A1 += euc_A1
                # Count colluding occurrences if cosine similarity > 0.8
                if cos_sim_A1 > 0.9:
                    collud_count += 1

                
            # Append total metrics for A1, D1 and D2 in order
            '''
            features_list[i]["features"].extend([
                cos_sim_tot_A1, euc_dist_tot_A1,
            ])
            '''
            
            colluding_scores[features_list[i]['cid']] = collud_count

        logger.info(f"Colluding scores: {colluding_scores}")
    
    # Proceed to clustering using the collected features.
    cids = []
    features = []
    for item in features_list:
        cids.append(item['cid'])
        features.append(item['features'])
    
    if features_list:
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
        cluster_labels = clusterer.fit_predict(X_scaled)
        clusters = {}
        for cid, label in zip(cids, cluster_labels):
            clusters.setdefault(label, []).append(cid)
        logger.info(f"Clusters formed: {clusters}")
        
        grad_dict = {cid: grad for cid, grad in gradients}
        cluster_means = {}
        for label, cid_list in clusters.items():
            updates = [grad_dict[cid]['flattened_diffs'] for cid in cid_list]
            updates_stack = torch.stack(updates, dim=0)
            cluster_means[label] = torch.mean(updates_stack, dim=0)
        
        # --- New: Compute cluster metrics and select best cluster ---
        cluster_scores = {}
        for label, cid_list in clusters.items():
            scores = []
            for cid in cid_list:
                # Find the corresponding feature vector for the client.
                client_feature = next(item for item in features_list if item["cid"] == cid)
                # Assuming feature order: [cosA, euclidean_A, server_cos, server_euc, ...]
                cosA = client_feature["features"][0]
                euclidean_A = client_feature["features"][1]
                server_cos = client_feature["features"][2]
                server_euc = client_feature["features"][3]
                #cos_tot = client_feature["features"][4]
                #euc_tot = client_feature["features"][5]
                score = cosA +  server_cos - server_euc - euclidean_A 
                scores.append(score)
            cluster_scores[label] = np.mean(scores)
            logger.info(f"Cluster {label} score: {cluster_scores[label]:.6f}")
            
        
        best_cluster = max(cluster_scores, key=cluster_scores.get)
        logger.info(f"Best cluster selected for aggregation: {best_cluster}")
        
        selected_cids = clusters.get(best_cluster, [])
        filtered_gradients = [grad for grad in gradients if grad[0] in selected_cids]
    else:
        filtered_gradients = gradients

    # Aggregate using FedAvg with filtered gradients
    global_update = fedavg(filtered_gradients, net, lr, f, device, **kwargs)
    
    # Exponential moving average (EMA) update of global update.
    ema_alpha = 0.3
    if server is not None:
        if server.last_global_update is None:
            server.last_global_update = global_update
        else:
            server.last_global_update = ema_alpha * global_update + (1 - ema_alpha) * server.last_global_update

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
            if sim > 0.92:
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