import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch.nn.functional as F

import numpy as np

from fl_simulator.utils.utility import segmentation
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
            param_update = global_update[idx:idx +
                                         param_length].reshape(param.size()).to(device)
            param.add_(param_update)
            idx += param_length
    logger.info("Global model parameters updated.")
