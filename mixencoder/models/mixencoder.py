import torch.nn as nn
import torch
from mixencoder.utils import mixer
from mixencoder.train import Trainer




class MixEncoder(nn.Module, Trainer):
    def __init__(self, input_size: int = 27, 
                 hidden_size: int = 10, 
                 emb_size: int = 10, 
                 enc_layers: int = 4, 
                 mix_layers: int = 2, 
                 restore_layers: int = 2, 
                 mode: str = "xmix", 
                 alpha: float = 5, 
                 beta: float = 2,
                 u_thresh: float = 0.9, 
                 l_thresh: float = 0.5):
        
        """
        MixEncoder model for pre-training.
        Args:
            input_size (int): Number of features
            hidden_size (int): Hidden layer size
            emb_size (int): Embedding size
            enc_layers (int): Number of layers in the encoder stack
            mix_layers (int): Number of layers in the mixing stack
            restore_layers (int): Number of layers in the restoration stack
            mode (str): Mode of the model - zmix, xmix (xmix for input mixing, zmix for latent mixing and rep. restoration)
            alpha (float): Alpha parameter for mixing
            beta (float): Beta parameter for mixing
            u_thresh (float): Upper threshold for mixing
            l_thresh (float): Lower threshold for mixing
        """
        
        super(MixEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.enc_layers = enc_layers
        self.mix_layers = mix_layers
        self.restore_layers = restore_layers
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.u_thresh = u_thresh
        self.l_thresh = l_thresh

        self._validate_config()

        print("\nSetting up Mix Encoder for pre-training...\n")
        print(f"Mode: {mode}\n")
    
        self.encoder_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            *[layer for _ in range(enc_layers-2) for layer in (nn.Linear(hidden_size, hidden_size), nn.GELU())],
            nn.Linear(hidden_size, emb_size),
            nn.GELU()
        )

        self.mix_stack = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            *[layer for _ in range(mix_layers-2) for layer in (nn.Linear(hidden_size, hidden_size), nn.GELU())],
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.restore_stack = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            *[layer for _ in range(restore_layers-2) for layer in (nn.Linear(hidden_size, hidden_size), nn.GELU())],
            nn.Linear(hidden_size, input_size if mode == "xmix" else emb_size)
        )

    def forward(self, x):
        if self.mode == "xmix":
            mix, lamb = mixer(x, self.alpha, self.beta, self.u_thresh, self.l_thresh)
            z = self.encoder_stack(mix)
            mix_pred = self.mix_stack(z)
            rest_pred = self.restore_stack(z)
            output = x
        elif self.mode == "zmix":
            z = self.encoder_stack(x)
            mix, lamb = mixer(z, self.alpha, self.beta, self.u_thresh, self.l_thresh)
            mix_pred = self.mix_stack(mix)
            rest_pred = self.restore_stack(mix)
            output = z
        out = {
            "rest_pred" : rest_pred,
            "mix_pred" : mix_pred,
            "lambda" : lamb,
            "z" : z,
            "mix" : mix,
            "output" : output
        }
        return out
    
    def _validate_config(self):
        assert self.mode in ["zmix", "xmix"], "Mode must be either 'zmix' or 'xmix'"
        assert isinstance(self.input_size, int) and self.input_size > 0, "input_size must be a positive integer."
        assert self.u_thresh > 0 and self.u_thresh < 1, "u_thresh must be between 0 and 1."
        assert self.l_thresh > 0 and self.l_thresh < 1 and self.l_thresh < self.u_thresh, "l_thresh must be between 0 and 1 and smaller than u_thresh."
        assert self.enc_layers >= 2, "enc_layers must be at least 2."
        assert self.mix_layers >= 2, "mix_layers must be at least 2."
        assert self.restore_layers >= 2, "restore_layers must be at least 2."
