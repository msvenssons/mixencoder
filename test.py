from models import MixEncoder

model = MixEncoder(mix_layers=5)

print(model.enc_layers)

# create test dataset using torch tensors
import numpy as np
import torch

X = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y = torch.tensor(np.random.rand(100).astype(np.float32))


X2 = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y2 = torch.tensor(np.random.rand(100).astype(np.float32))


model.fit(X, y, X2, y2)

