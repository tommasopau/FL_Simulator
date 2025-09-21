import sklearn
from sklearn.neighbors import KernelDensity
import numpy as np
import json
import yaml
import scipy.signal as signal
from scipy.signal import argrelextrema
from sklearn.cluster import estimate_bandwidth
from typing import Dict



def segmentation(trust_scores: np.ndarray, kernel : str) -> float:
    """
    Segments the trust scores into clusters based on density estimation and identifies the last cluster boundary.
    
    Args:
        trust_scores (np.ndarray): Array of trust scores.
    
    Returns:
        float: The last cluster boundary or the minimum trust score if no cluster boundaries are found.
    """
    bandwidth = sklearn.cluster.estimate_bandwidth(trust_scores.reshape(-1,1))
    print(f"Bandwidth : {bandwidth}")

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(trust_scores.reshape(-1,1))
        
    # Generate density values for the trust scores
    x_d = np.linspace(trust_scores.min() - 1, trust_scores.max() + 1, 100000)
    log_density = kde.score_samples(x_d.reshape(-1, 1))
    density = np.exp(log_density)
        
    # Find local minima in the density to identify cluster boundaries
    minima_indices = argrelextrema(density, np.less)[0]
    cluster_boundaries = x_d[minima_indices]
    print(f"Clusters : {cluster_boundaries}")
    if len(cluster_boundaries) == 0:
        return min(trust_scores)
    last_segment = cluster_boundaries[-1]  
      
    return last_segment




        