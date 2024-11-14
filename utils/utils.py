import numpy as np
import torch

def mixer(x, alpha, beta, u_thresh, l_thresh): 
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