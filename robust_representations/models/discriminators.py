import numpy as np
import torch
import torch.nn as nn


class PriorDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(64, 1000),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(1000, 200),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1),
            nn.ReLU(),
        )
        self.layers = [self.layer0, self.layer1, self.layer2]

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        def _clip_grad(v, min, max):
            v_tmp = v.expand_as(v)
            v_tmp.register_hook(lambda g: g.clamp(min, max))
            return v_tmp

        out = []
        for layer in self.layers:
            x = layer(x)
            if clip_grad:
                x = _clip_grad(x, -clip_grad, clip_grad)
            out.append(x)

        if not return_full_list:
            out = out[-1]

        return out


class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units,):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.fcnet = nn.Sequential(
            # nn.Linear(2 * 512 * 30 + num_classes, 4096),
            nn.Linear(n_input, 64),
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_units)
        )

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """

        h = self.fcnet(x.float())
        return h


class MIFCNet(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units, bn =False):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        """

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: network output.

        """


        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)

        return h


class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)