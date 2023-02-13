import torch
from torch import nn


class BasicEncoder(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(9, 10),
            nn.BatchNorm1d(10),  # normalize
            nn.ReLU(inplace=True),
            nn.Linear(10, 3)
        )

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False,
                prop_limit=None):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''
        out = self.net1(x.float())
        return out


class View(torch.nn.Module):
    """Basic reshape module.

    """
    def __init__(self, *shape):
        """

        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: Flattened tensor.

        """
        return input.view(*self.shape)


basic_encoder = BasicEncoder