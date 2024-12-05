import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from mixencoder.train import Trainer



class MixSetup(Trainer):
    def _setup(self):
        
        setup = {
        "losses" : {
            "rest_loss" : {
                "type" : nn.MSELoss(),
                "weight" : 1.0,
                "pred" : "rest_pred",
                "target" : "z"
            },
            "mix_loss" : {
                "type" : nn.MSELoss(),
                "weight" : 0.3,
                "pred" : "mix_pred",
                "target" : "lambda"
            }
        },
    }
        return setup