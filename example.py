from mixencoder.models import MixEncoder
from mixencoder.utils import FineTuner
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("example_data.csv", sep='\t')
X = df.drop(columns=["target"])
y = df[["target"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values.astype(np.float32))
y_train = torch.tensor(y_train.astype(np.float32))

X_test = torch.tensor(X_test.values.astype(np.float32))
y_test = torch.tensor(y_test.astype(np.float32))

model = MixEncoder(input_size=X_train.shape[1], mode="xmix")
model.fit(X_train, y_train, X_test, y_test, plot=True, epochs=100)

finetuner = FineTuner(model)
finetuner.finetune(X_train, y_train, X_test, y_test, plot=True, epochs=100, mode="cls", metric="AUC")