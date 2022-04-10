import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow.act_norm import ActNorm
from models.glow.coupling import Coupling
from models.glow.inv_conv import InvConv

import numpy as np


class Glow(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        num_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, num_channels, num_levels, num_steps, quants=256, num_image_channels=1):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * num_image_channels,  # RGB image after squeeze
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)

        self.dequantization = Dequantization(quants=quants)

    def forward(self, x, reverse=False, quant_int=True):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            sldj = torch.zeros(x.size(0), device=x.device)
            x, sldj = self.dequantization(x, sldj)

        x = squeeze(x)
        x, sldj = self.flows(x, sldj, reverse)
        x = squeeze(x, reverse=True)

        if reverse:
            x, sldj = self.dequantization(x, sldj, reverse=True, quant_int=quant_int)

        return x, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x

class Dequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=256, add_unif_noise=False):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants
        self.add_unif_noise = add_unif_noise

    def forward(self, z, ldj, reverse=False, quant_int=True):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            if quant_int:
                z = torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        if self.add_unif_noise:
            z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj

# class VariationalDequantization(Dequantization):

#     def __init__(self, var_flows, alpha=1e-5):
#         """
#         Inputs:
#             var_flows - A list of flow transformations to use for modeling q(u|x)
#             alpha - Small constant, see Dequantization for details
#         """
#         super().__init__(alpha=alpha)
#         self.flows = nn.ModuleList(var_flows)

#     def dequant(self, z, ldj):
#         z = z.to(torch.float32)
#         img = (z / 255.0) * 2 - 1 # We condition the flows on x, i.e. the original image

#         # Prior of u is a uniform distribution as before
#         # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
#         deq_noise = torch.rand_like(z).detach()
#         deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
#         for flow in self.flows:
#             deq_noise, ldj = flow(deq_noise, ldj, reverse=False, orig_img=img)
#         deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

#         # After the flows, apply u as in standard dequantization
#         z = (z + deq_noise) / 256.0
#         ldj -= np.log(256.0) * np.prod(z.shape[1:])
#         return z, ldj
