import torch 
import numpy as np
from typing import List



def min_max_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning).
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    if dev_type == 'unit_vec':
        deviation = grad_mean / torch.norm(grad_mean, p=2)  # unit vector, dir opposite to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(grad_mean)  # signs of the gradients
    elif dev_type == 'std':
        deviation = torch.std(catv, dim=1)  # standard deviation
    else:
        raise ValueError("Invalid dev_type. Choose from 'unit_vec', 'sign', or 'std'.")
    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)  # determine max distance left side of optimization
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)

    for i in range(num_attackers_epoch):
        v[i] = mal_update[:, None]
    
        
        

    return v



def min_sum_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning).
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
    """
    
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    if dev_type == 'unit_vec':
        deviation = grad_mean / torch.norm(grad_mean, p=2)  # unit vector, dir opposite to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(grad_mean)  # signs of the gradients
    elif dev_type == 'std':
        deviation = torch.std(catv, dim=1)  # standard deviation
    else:
        raise ValueError("Invalid dev_type. Choose from 'unit_vec', 'sign', or 'std'.")


    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)

    for i in range(num_attackers_epoch):
        v[i] = mal_update[:, None]
    return v
    
    
def krum_attack(
    v: List[torch.Tensor],
    net: torch.nn.Module,
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Local model poisoning attack against the Krum aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        net (torch.nn.Module): Model.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
    """
    threshold = 1e-5

    n = len(v)
    w_re = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0)
    d = v[0].size()[0]
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):  # compute euclidean distance of benign to benign devices
        for j in range(i + 1, n):
            d = torch.norm(lr * v[i] - lr * v[j], p=2)
            dist[i, j], dist[j, i] = d, d

    dist_benign_sorted, _ = torch.sort(dist[f:, f:])
    min_dist = torch.min(torch.sum(dist_benign_sorted[:, 0:(n - f - 1)], dim=-1))

    dist_w_re = []
    for i in range(f, n):
        dist_w_re.append(torch.norm(lr * v[i], p=2))
    max_dist_w_re = torch.max(torch.stack(dist_w_re))

    max_lambda = min_dist / ((n - 2 * f - 1) * torch.sqrt(d)) + max_dist_w_re / torch.sqrt(d)

    actual_lambda = max_lambda
    sorted_dist, _ = torch.sort(dist, dim=-1)
    update_before = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    while actual_lambda > threshold:
        for i in range(num_attackers_epoch):
            v[i] = - actual_lambda * torch.sign(update_before)

        dist = torch.zeros((n, n)).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(v[i] - v[j])
                dist[i, j], dist[j, i] = d, d
        sorted_dist, _ = torch.sort(dist, dim=-1)
        global_update = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
        if torch.equal(global_update, v[0]):
            break
        else:
            actual_lambda = actual_lambda / 2

    return v

def trim_attack(
    v: List[torch.Tensor],
    net: torch.nn.Module,
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec'
) -> List[torch.Tensor]:
    """
    Local model poisoning attack using the trim method.
    
    Parameters:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients, where the first f are malicious.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Device used in training and inference.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
    """
    vi_shape = tuple(v[0].size())
    v_tran = torch.cat(v, dim=1)
    maximum_dim, _ = torch.max(v_tran, dim=1, keepdim=True)
    minimum_dim, _ = torch.min(v_tran, dim=1, keepdim=True)
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range( num_attackers_epoch):
        random_12 = (1. + torch.rand(*vi_shape)).to(device)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v

def no_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
    ) -> List[torch.Tensor]:
    """
    No attack.
    
    Parameters:
        v (List[torch.Tensor]): List of gradients.
        
    """
    return v

def label_flip_attack (v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
    ) -> List[torch.Tensor]:
    """
    Label flip attack. It flips the labels of the first f clients but it happens before the training.
    Hence here we do not need to change the gradients.
    """
    return v

def gaussian_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Malicious attack in which each malicious client's update is replaced by a randomly 
    generated vector sampled from a Gaussian distribution with a mean of 0 and a variance of 200.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate (unused).
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates to generate.
        device (torch.device): Computation device.
    
    Returns:
        List[torch.Tensor]: The list of gradients with malicious updates applied.
    """
    # Standard deviation is sqrt(200) ~ 14.1421.
    std = torch.sqrt(torch.tensor(200.0)).item()
    for i in range(num_attackers_epoch):
        v[i] = torch.normal(mean=0.0, std=std, size=v[i].size()).to(device)
    return v

import torch.nn.functional as F

def min_max_attack_variant(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec',
    noise_std: float = 0.01,  # initial standard deviation for the noise
    cos_threshold: float = 0.90  # desired minimum cosine similarity
) -> List[torch.Tensor]:
    """
    Variant of the min max attack which adds a small noise to each malicious update vector.
    The first attacker (index 0) will receive the exact crafted update, while subsequent attackers 
    receive a version with added noise. The noise is scaled to guarantee that the cosine similarity 
    with the original update is at least cos_threshold.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates per epoch.
        device (torch.device): Device used in training.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
        noise_std (float, optional): Initial standard deviation for the Gaussian noise. Defaults to 0.01.
        cos_threshold (float, optional): Minimum cosine similarity threshold. Defaults to 0.90.
    
    Returns:
        List[torch.Tensor]: List of gradients with malicious (and partly noisy) updates.
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    
    if dev_type == 'unit_vec':
        deviation = grad_mean / torch.norm(grad_mean, p=2)
    elif dev_type == 'sign':
        deviation = torch.sign(grad_mean)
    elif dev_type == 'std':
        deviation = torch.std(catv, dim=1)
    else:
        raise ValueError("Invalid dev_type. Choose from 'unit_vec', 'sign', or 'std'.")

    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    # Finding optimal gamma as in algorithm 1.
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    
    mal_update = (grad_mean - gamma_succ * deviation)
    
    # First attacker gets the update without noise.
    v[0] = mal_update[:, None]

    # Other attackers receive a noisy version.
    for i in range(1, num_attackers_epoch):
        noise = torch.normal(mean=0.0, std=noise_std, size=mal_update.size()).to(device)
        noisy_update = mal_update + noise
        cos_sim = F.cosine_similarity(mal_update, noisy_update, dim=0).item()
        # Scale down the noise until cosine similarity is at least `cos_threshold`.
        while cos_sim < cos_threshold:
            noise = noise * 0.5
            noisy_update = mal_update + noise
            cos_sim = F.cosine_similarity(mal_update, noisy_update, dim=0).item()
        v[i] = noisy_update[:, None]
        print(f"Attacker {i}: cosine similarity = {cos_sim:.4f}")

    return v

def sign_flip_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    An attack that flips the sign of the updates for the malicious clients.

    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate (unused).
        f (int): Number of malicious clients (unused, kept for consistency).
        num_attackers_epoch (int): Number of malicious updates to flip.
        device (torch.device): Device used in training.

    Returns:
        List[torch.Tensor]: List of gradients with sign-flipped updates for the malicious clients.
    """
    for i in range(num_attackers_epoch):
        v[i] = -v[i]
    return v