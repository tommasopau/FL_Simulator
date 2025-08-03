import torch
from torch import nn
from typing import List, Dict, Optional, Tuple
import logging
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from app.aggregation_techniques.fedavg import fedavg

logger = logging.getLogger(__name__)


def cluster_similarity_fedavg(
    gradients: List[Tuple[int, Dict[str, torch.Tensor]]],
    net: nn.Module,
    lr: float,
    f: int,
    device: torch.device,
    **kwargs
) -> None:
    """
    Simple cluster-aware aggregation that logs cosine similarities within clusters 
    and then applies standard FedAvg.

    Args:
        gradients: List of tuples containing client ID and their gradients
        net: The global model to be updated
        lr: Learning rate
        f: Number of malicious clients
        device: Computation device
        clusters: Dictionary mapping client_id -> cluster_id
        **kwargs: Additional keyword arguments
    """
    logger.info("Aggregating gradients using Cluster-Similarity FedAvg method.")
    clusters = kwargs.get('clusters', None)
    if clusters is None:
        logger.warning(
            "No cluster information provided. Using standard FedAvg.")
        fedavg(gradients, net, lr, f, device, **kwargs)
        return

    # Group clients by cluster
    cluster_groups = defaultdict(list)
    for cid, gradient in gradients:
        # Default to -1 if client not in clusters
        cluster_id = clusters.get(cid, -1)
        cluster_groups[cluster_id].append((cid, gradient))

    logger.info(
        f"Grouped {len(gradients)} clients into {len(cluster_groups)} clusters")

    # Log cosine similarities within each cluster - only negative ones
    negative_similarity_clients = []

    for cluster_id, cluster_clients in cluster_groups.items():
        if len(cluster_clients) < 2:
            continue

        # Calculate pairwise cosine similarities
        for i, (cid1, grad1) in enumerate(cluster_clients):
            flat_grad1 = grad1['flattened_diffs'].view(-1)

            for j, (cid2, grad2) in enumerate(cluster_clients):
                if i < j:  # Only calculate upper triangle to avoid duplicates
                    flat_grad2 = grad2['flattened_diffs'].view(-1)
                    similarity = F.cosine_similarity(
                        flat_grad1, flat_grad2, dim=0).item()

                    # Only log negative similarities
                    if similarity < 0:
                        logger.warning(
                            f"NEGATIVE SIMILARITY in Cluster {cluster_id}: Client {cid1} and Client {cid2}: {similarity:.4f}")
                        if cid1 not in negative_similarity_clients:
                            negative_similarity_clients.append(cid1)
                        if cid2 not in negative_similarity_clients:
                            negative_similarity_clients.append(cid2)

    # Log summary of clients with negative similarities
    if negative_similarity_clients:
        logger.warning(
            f"Clients with negative cosine similarities: {sorted(negative_similarity_clients)}")
    else:
        logger.info("No negative cosine similarities detected within clusters")

    # Apply standard FedAvg aggregation
    logger.info("Applying standard FedAvg aggregation after similarity analysis")
    fedavg(gradients, net, lr, f, device, **kwargs)
