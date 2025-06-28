import torch 
from typing import List


def sign_flip_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Performs an attack that flips the sign of the updates for the malicious clients.

    Args:
        v (List[torch.Tensor]): List of client gradients.
        lr (float): Learning rate (unused).
        f (int): Number of malicious clients (unused, kept for compatibility).
        num_attackers_epoch (int): Number of malicious updates to flip.
        device (torch.device): Computation device used during training.

    Returns:
        List[torch.Tensor]: List of gradients with sign-flipped updates for the malicious clients.
    """
    for i in range(num_attackers_epoch):
        v[i] = -v[i]
    return v