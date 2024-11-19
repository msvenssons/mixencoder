import numpy as np
import torch

def mixer(x: torch.Tensor, alpha: float, beta: float, u_thresh: float, l_thresh: float) -> torch.Tensor: 
    """
    Mixer function for MixEncoder.
    Args:
        x (torch.Tensor): Input tensor
        alpha (float): Alpha parameter for Beta distribution
        beta (float): Beta parameter for Beta distribution
        u_thresh (float): Upper threshold for lambda
        l_thresh (float): Lower threshold for lambda
    """
    np.random.seed(1)
    torch.manual_seed(1)
    lamb = torch.distributions.Beta(alpha, beta).sample((x.shape[0], 1)).to(x.device)
    lamb = torch.round(lamb, decimals=2)
    lamb = torch.where(lamb < l_thresh, torch.tensor(l_thresh), lamb)
    lamb = torch.where(lamb > u_thresh, torch.tensor(1.), lamb)

    shuffled_x = torch.clone(x)
    shuffled_x = shuffled_x[torch.randperm(shuffled_x.size()[0])]
    mixed_x = (x*lamb) + (shuffled_x*(1-lamb))
    
    return mixed_x, lamb

# TODO: gridsearch for lambda