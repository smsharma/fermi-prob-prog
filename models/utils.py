import numpy as np
from torch import nn, tanh, relu
from pyro.distributions.transforms import affine_autoregressive

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms

def build_mlp(input_dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

def build_iaf(input_dim, num_iafs, iaf_dim):
    iafs = [affine_autoregressive(input_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
    return iafs

def build_maf(dim=1, num_transforms=8, context_features=None, hidden_features=128):

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=dim),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    distribution = distributions_.StandardNormal((dim,))
    neural_net = flows.Flow(transform, distribution)
    
    return neural_net

class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
    
def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.
    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.
    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd