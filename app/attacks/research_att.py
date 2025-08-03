import torch 
import numpy as np
from typing import List

import torch.nn.functional as F

def min_max_attack_rand_noise(
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

def min_sum_attack_rand_noise(
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
    Variant of the min sum attack which adds a small noise to each malicious update vector.
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
    gamma_fail = gamma.clone()
    gamma_succ = 0

    scores = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        scores.append(torch.sum(distance))
    min_score = torch.min(torch.stack(scores))
    
    # Finding optimal gamma according to algorithm 1 (min sum version).
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = grad_mean - gamma * deviation
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)
        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2
        gamma_fail = gamma_fail / 2
    
    mal_update = grad_mean - gamma_succ * deviation
    
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
        
    
    return v


import numpy as np

def perturb_attack(attack: np.ndarray, client_id: int, cosine_threshold: float = 0.9) -> np.ndarray:
    """
    Perturb the attack update ensuring at least `cosine_threshold` cosine similarity.
    
    Parameters:
      attack (np.ndarray): The original attack vector.
      client_id (int): A unique identifier for the client (used for deterministic seeding).
      cosine_threshold (float): Minimum cosine similarity required.
      
    Returns:
      np.ndarray: The perturbed attack vector.
    """
    # Initialize a deterministic random generator using client_id as seed.
    rng = np.random.RandomState(seed=client_id)
    
    # Generate a random noise vector with the same shape as the attack vector.
    noise = rng.randn(*attack.shape)
    
    # Remove the component of the noise that is along the attack vector (make it orthogonal).
    proj = (np.dot(noise, attack) / np.dot(attack, attack)) * attack
    noise -= proj
    
    # Normalize the noise vector.
    noise_norm = np.linalg.norm(noise)
    if noise_norm == 0:
        return attack.copy()
    noise = noise / noise_norm
    
    # Calculate the maximum noise magnitude allowed for the cosine similarity threshold.
    max_noise_magnitude = np.linalg.norm(attack) * np.sqrt(1 / (cosine_threshold ** 2) - 1)
    
    # Scale the noise to this maximum magnitude.
    noise = noise * max_noise_magnitude
    
    # Construct the perturbed attack vector.
    perturbed_attack = attack + noise
    
    return perturbed_attack


def min_max_attack_ortho(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec',
    cos_threshold: float = 0.90
) -> List[torch.Tensor]:
    """
    A variant of the min max attack which, after computing the crafted update,
    perturbs each malicious update (except for the first) using deterministic noise.
    The noise is generated using the `perturb_attack` function ensuring that
    the cosine similarity with the clean update is at least `cos_threshold`.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates per epoch.
        device (torch.device): Device used in training.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
        cos_threshold (float, optional): Minimum cosine similarity threshold. Defaults to 0.90.
    
    Returns:
        List[torch.Tensor]: List of gradients with malicious (and perturbed) updates.
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
    gamma_fail = gamma.clone()
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    max_distance = torch.max(distances)
    del distances

    # Find optimal gamma as in algorithm 1.
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = grad_mean - gamma * deviation
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2

    mal_update = grad_mean - gamma_succ * deviation

    # First attacker gets the clean update.
    v[0] = mal_update[:, None]

    # Subsequent attackers receive a perturbed update using the helper function.
    for i in range(1, num_attackers_epoch):
        # Convert the crafted update to a numpy array.
        mal_update_np = mal_update.detach().cpu().numpy()
        # Generate the perturbed attack using the deterministic procedure.
        perturbed_np = perturb_attack(mal_update_np, client_id=i, cosine_threshold=cos_threshold)
        perturbed = torch.tensor(perturbed_np, device=device, dtype=mal_update.dtype)
        v[i] = perturbed[:, None]
        cos_sim = F.cosine_similarity(mal_update, perturbed, dim=0).item()
        print(f"Attacker {i}: cosine similarity = {cos_sim:.4f}")

    return v

def min_sum_attack_ortho(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device,
    dev_type: str = 'unit_vec',
    cos_threshold: float = 0.90
) -> List[torch.Tensor]:
    """
    A variant of the min sum attack which, after computing the crafted update,
    perturbs each malicious update (except for the first) using deterministic noise.
    The noise is generated using the `perturb_attack` function ensuring that
    the cosine similarity with the clean update is at least `cos_threshold`.
    
    Args:
        v (List[torch.Tensor]): List of gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of malicious updates per epoch.
        device (torch.device): Device used in training.
        dev_type (str, optional): Type of deviation ('unit_vec', 'sign', 'std'). Defaults to 'unit_vec'.
        cos_threshold (float, optional): Minimum cosine similarity threshold. Defaults to 0.90.
    
    Returns:
        List[torch.Tensor]: List of gradients with malicious (and perturbed) updates.
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
    gamma_fail = gamma.clone()
    gamma_succ = 0

    scores = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        scores.append(torch.sum(distance))
    min_score = torch.min(torch.stack(scores))
    
    # Finding optimal gamma according to min sum criterion.
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = grad_mean - gamma * deviation
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)
        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2
        gamma_fail = gamma_fail / 2

    mal_update = grad_mean - gamma_succ * deviation

    # First attacker gets the clean update.
    v[0] = mal_update[:, None]

    # Subsequent attackers receive a perturbed update using the helper function.
    for i in range(1, num_attackers_epoch):
        # Convert the crafted update to a numpy array.
        mal_update_np = mal_update.detach().cpu().numpy()
        # Generate the perturbed attack using the deterministic procedure.
        perturbed_np = perturb_attack(mal_update_np, client_id=i, cosine_threshold=cos_threshold)
        perturbed = torch.tensor(perturbed_np, device=device, dtype=mal_update.dtype)
        v[i] = perturbed[:, None]
        cos_sim = F.cosine_similarity(mal_update, perturbed, dim=0).item()
        print(f"Attacker {i}: cosine similarity = {cos_sim:.4f}")

    return v