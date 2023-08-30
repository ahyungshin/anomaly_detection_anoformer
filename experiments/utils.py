import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

def seed_all(seed: int = 8):

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class SoftArgmax2D(nn.Module):
    """Creates a module that computes Soft-Argmax 2D of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y].
    """
    def __init__(self, beta: int = 1000, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, vocab = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, vocab), dim=2
        ).view(batch_size, num_channel, vocab)

        xx, yy = torch.meshgrid(list(map(torch.arange, [num_channel, vocab])))
        yy = yy.repeat(batch_size, 1, 1)

        approx_x = (
            softmax.mul(yy.float().to(device))
            .view(batch_size, num_channel, vocab)
            .sum(2)
            .unsqueeze(2)
        )

        output = approx_x
        return output
