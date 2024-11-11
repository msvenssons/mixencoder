import torch.nn as nn
import torch
from utils import mixer
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class MixEncoder(nn.Module):
    def __init__(self, input_size: int = 27, hidden_size: int = 10, emb_size: int = 10, 
                 enc_layers: int = 4, mix_layers: int = 2, restore_layers: int = 2, mode: str = "zmix", lamb: float = 0.5):
        super(MixEncoder, self).__init__()

        # initialization
        self.mode = mode
        self.lamb = lamb

        print("Setting up Mix Encoder for pre-training...\n")
        print(f"Mode: {mode}\n")
    
        # TODO: different hidden sizes for different layers
        self.encoder_stack = nn.ModuleList(
            [nn.Linear(input_size, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(enc_layers-2)] +
            [nn.Linear(hidden_size, emb_size)]
        )

        self.mix_stack = nn.ModuleList(
            [nn.Linear(emb_size, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(mix_layers-2)] +
            [nn.Linear(hidden_size, 1)]
        )

        self.restore_stack = nn.ModuleList(
            [nn.Linear(emb_size, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(enc_layers-2)] +
            [nn.Linear(hidden_size, input_size)]
        )

    def forward(self, x):
        if self.mode == "xmix":
            mix = mixer(x, self.lamb)
            z = self.encoder_stack(mix)
            mix_pred = self.mix_stack(z)
            rest_pred = self.restore_stack(z)
        elif self.mode == "zmix":
            z = self.encoder_stack(x)
            mix = mixer(z, self.lamb)
            mix_pred = self.mix_stack(mix)
            rest_pred = self.restore_stack(mix)
        out = {
            "rest_pred" : rest_pred,
            "mix_pred" : mix_pred,
            "lambda" : self.lamb,
            "z" : z,
            "mix" : mix
        }
        return out
    
    def _validate_config(self):
        # Validate input_size
        if self.input_size <= 0 or self.input_size.type != int:
            raise ValueError("input_size must be a positive integer.")

        # Validate enc_layers
        if self.enc_layers < 2:
            raise ValueError("enc_layers must be at least 2.")

        # Validate mixing_type
        allowed_modes = ["zmix", "xmix"]
        if self.mode not in allowed_modes:
            raise ValueError(
                f"Invalid 'mixing_type': '{self.mode}'. "
                f"Allowed values are {allowed_modes}."
            )
        # Additional checks can go here
