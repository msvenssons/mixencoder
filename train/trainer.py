# add a basemodel with a train class + method and then have mixencoder inherit from it
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import tqdm
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()


        self.train_metric = None
        self.val_metric = None


    def fit(self, x_train, y_train, x_val, y_val, mode: str = 'cls', metric: str = "accuracy", epochs: int = 100, batch_size: int = 200, lr: float = 1e-3, device: str = "cpu"):
        # TODO: add testing functionality
        
        """Training loop for mixencoder.
        Args:
            x_train (torch.Tensor): Training data
            y_train (torch.Tensor): Training labels
            x_val (torch.Tensor): Validation data
            y_val (torch.Tensor): Validation labels
            train_data (torch.Tensor): Training data
            val_data (torch.Tensor): Validation data
            mode (str): Mode of the model - cls, mcls, reg
            metric (str): Metric to evaluate - accuracy, AUC, MSE
            epochs (int): Number of epochs
            batch_size (int): Batch size
            lr (float): Learning rate
            device (str): Device to train on - cpu, cuda
        """

        
        # assertions for input validation

        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'"
        assert mode in ["cls", "mcls", "reg"], "Mode must be either 'cls', 'mcls' or 'reg'"
        assert metric in ["accuracy", "AUC", "MSE"], "Invalid metric - [accuracy, AUC, MSE]"
        assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same number of samples"
        assert x_val.shape[0] == y_val.shape[0], "x_val and y_val must have the same number of samples"
        assert isinstance(x_train, torch.Tensor), "x_train must be a torch.Tensor"
        assert isinstance(y_train, torch.Tensor), "y_train must be a torch.Tensor"
        assert isinstance(x_val, torch.Tensor), "x_val must be a torch.Tensor"
        assert isinstance(y_val, torch.Tensor), "y_val must be a torch.Tensor"



        # set device and criterion

        if device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("cuda not available; using cpu")
        if mode == "cls":
            criterion = nn.BCELoss()
        elif mode == "mcls":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()


        # set training parameters

        self.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


        # main training and validation loop

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for i, data in enumerate(tqdm.tqdm(train_loader)): # bug; will probably fill the terminal with new bars
                optimizer.zero_grad()
                x, y = data
                #x, y = x.to(device), y.to(device)
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")

            if (epoch + 1) % 10 == 0:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for i, data in enumerate(tqdm.tqdm(val_loader)):
                        x, y = data
                        #x, y = x.to(device), y.to(device)
                        y_pred = self(x)
                        loss = criterion(y_pred, y)
                        val_loss += loss.item()
                    val_loss /= len(val_loader)
                    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss}")
        

        # calculate and print metrics

        self._evaluate_metrics(self, train_loader, val_loader, metric, device)


    def _evaluate_metrics(self, model, train_loader, val_loader, metric, device):
        """Evaluate metrics on training and validation data."""
        
        model.eval()
        train_true, train_pred = [], []
        val_true, val_pred = [], []

        with torch.no_grad():
            for data in train_loader:
                x, y = data
                #x, y = x.to(device), y.to(device)
                y_pred = model(x)
                train_true.extend(y.cpu().numpy())
                train_pred.extend(y_pred.cpu().numpy())

            for data in val_loader:
                x, y = data
                #x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_true.extend(y.cpu().numpy())
                val_pred.extend(y_pred.cpu().numpy())

        train_metric = self._calculate_metric(metric, train_true, train_pred)
        val_metric = self._calculate_metric(metric, val_true, val_pred)

        self.train_metric = train_metric
        self.val_metric = val_metric

        print(f"Final Train {metric}: {train_metric}")
        print(f"Final Val {metric}: {val_metric}")


    def _calculate_metric(self, metric, y_true, y_pred):
        """Calculate metric."""

        if metric == "accuracy":
            y_pred = (y_pred > 0.5).astype(int)
            return (y_true == y_pred).mean()
        elif metric == "AUC":
            return roc_auc_score(y_true, y_pred)
        elif metric == "MSE":
            return ((y_true - y_pred) ** 2).mean()
        else:
            raise ValueError("Invalid metric - [accuracy, AUC, MSE]")
        
    def test(self):
        print("testing")
        pass
