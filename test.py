from mixencoder.models import MixEncoder
from mixencoder.utils import FineTuner
import numpy as np
import torch

"""model = MixEncoder(mix_layers=5, mode="xmix")

print(model.enc_layers)

X = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y = torch.tensor(np.random.rand(100).astype(np.float32))


X2 = torch.tensor(np.random.rand(1000, 27).astype(np.float32))
y2 = torch.tensor(np.random.rand(1000).astype(np.float32))

# create classification data

X = torch.tensor(np.random.rand(1000, 27).astype(np.float32))
y = torch.tensor(np.random.randint(0, 2, 1000).astype(np.float32))

X2 = torch.tensor(np.random.rand(100, 27).astype(np.float32))
y2 = torch.tensor(np.random.randint(0, 2, 100).astype(np.float32))


model.fit(X, y, X2, y2, plot=True, epochs=100)

finetuner = FineTuner(model)
finetuner.finetune(X, y, X2, y2, plot=True, epochs=100, mode="cls", metric="accuracy")"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("N0_data.csv", sep='\t')
X = df.drop(columns=["N0", "ID"])
y = df[["N0"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values.astype(np.float32))
y_train = torch.tensor(y_train.astype(np.float32))

X_test = torch.tensor(X_test.values.astype(np.float32))
y_test = torch.tensor(y_test.astype(np.float32))

model = MixEncoder(input_size=X_train.shape[1], mode="xmix")
model.fit(X_train, y_train, X_test, y_test, plot=True, epochs=100)

finetuner = FineTuner(model)
finetuner.finetune(X_train, y_train, X_test, y_test, plot=True, epochs=100, mode="cls", metric="AUC")