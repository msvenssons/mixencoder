import torch.nn as nn

class PredictionHead(nn.Module):
    """Prediction Head for Mix Encoder; a simple feed forward MLP.
    """
    def __init__(self, 
                 input_size:int = 10, 
                 linear: bool = True, 
                 mode: str = "cls"):
        
        self.mode = mode
        
        super(PredictionHead, self).__init__()

        if linear:
            self.prediction_stack = nn.Sequential(
                nn.Linear(input_size, 1)
            )
        else:
            self.prediction_stack = nn.Sequential(
                nn.Linear(input_size, input_size//2),
                nn.GELU(),
                nn.Linear(input_size//2, 1)
            )


    def forward(self, x):
        if self.mode == "cls":
            s = nn.Sigmoid()
            out = s(self.prediction_stack(x))
        elif self.mode == "mcls":
            m = nn.Softmax(dim=1)
            out = m(self.prediction_stack(x))
        else:
            out = self.prediction_stack(x)
        return out