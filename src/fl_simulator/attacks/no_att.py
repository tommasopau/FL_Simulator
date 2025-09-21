import torch
import numpy as np
from typing import List


def no_attack(
    v: List[torch.Tensor],
    lr: float,
    f: int,
    num_attackers_epoch: int,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Applies no attack.

    The gradients remain unchanged.

    Args:
        v (List[torch.Tensor]): List of client gradients.
        lr (float): Learning rate.
        f (int): Number of malicious clients.
        num_attackers_epoch (int): Number of attackers per epoch.
        device (torch.device): Computation device.

    Returns:
        List[torch.Tensor]: Unmodified list of gradients.
    """
    return v