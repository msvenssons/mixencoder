from models import MixEncoder
from utils import FineTuner
import numpy as np
import torch

model = MixEncoder(mix_layers=5, mode="xmix")

print(model.enc_layers)

"""X = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y = torch.tensor(np.random.rand(100).astype(np.float32))


X2 = torch.tensor(np.random.rand(1000, 27).astype(np.float32))
y2 = torch.tensor(np.random.rand(1000).astype(np.float32))"""

# create classification data

X = torch.tensor(np.random.rand(1000, 27).astype(np.float32))
y = torch.tensor(np.random.randint(0, 2, 1000).astype(np.float32))

X2 = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y2 = torch.tensor(np.random.randint(0, 2, 100).astype(np.float32))


model.fit(X, y, X2, y2, plot=True, epochs=100)

finetuner = FineTuner(model)
finetuner.finetune(X, y, X2, y2, plot=True, epochs=100, mode="cls", metric="accuracy")