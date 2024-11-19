# add a basemodel with a train class + method and then have mixencoder inherit from it
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()


        self.train_metric = None
        self.val_metric = None
        self.device = "cpu"
        self.train_losses = []
        self.val_losses = []


    def fit(self, 
            x_train: torch.Tensor, 
            y_train: torch.Tensor, 
            x_val: torch.Tensor, 
            y_val: torch.Tensor, 
            mode: str = 'cls', 
            metric: str = "accuracy", 
            epochs: int = 100, 
            batch_size: int = 200, 
            lr: float = 1e-3,
            l_scale: float = 0.3,
            plot: bool = True, 
            device: str = "cpu"):
        
        # TODO: add testing functionality
        #       I don't need metrics here, this is just to pretrain the encoder
        
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
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("cuda not available; using cpu\n")
        if mode == "cls":
            criterion = nn.BCELoss()
        elif mode == "mcls":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()


        # set training parameters
        # TODO: modularize this so it can be reused in finetuner

        self.to(self.device)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)
        rest_criterion = nn.MSELoss()
        mix_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.train_losses, self.val_losses = [], []


        # main training and validation loop

        t = tqdm.tqdm(range(epochs))

        for epoch in t:
            self.train()
            train_rest_loss = 0
            train_mix_loss = 0
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                out = self(x)
                rest_loss = rest_criterion(x, out["rest_pred"])
                mix_loss = mix_criterion(out["lambda"], out["mix_pred"])
                total_loss = rest_loss + l_scale*mix_loss
                total_loss.backward()
                optimizer.step()
                train_rest_loss += rest_loss.item()
                train_mix_loss += mix_loss.item()
            train_rest_loss /= len(train_loader)
            train_mix_loss /= len(train_loader)
            self.train_losses.append((train_rest_loss, train_mix_loss))
            t.set_description(f"Train Epoch {epoch+1}/{epochs}")
            t.set_postfix_str(f"Train Rest Loss: {train_rest_loss:.4f}, Train Mix Loss: {train_mix_loss:.4f}", refresh=True)


            self.eval()
            val_rest_loss = 0
            val_mix_loss = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    out = self(x)
                    rest_loss = rest_criterion(x, out["rest_pred"])
                    mix_loss = mix_criterion(out["lambda"], out["mix_pred"])
                    val_rest_loss += rest_loss.item()
                    val_mix_loss += mix_loss.item()
                val_rest_loss /= len(val_loader)
                val_mix_loss /= len(val_loader)
                self.val_losses.append((val_rest_loss, val_mix_loss))
                t.set_description(f"Val Epoch {epoch+1}/{epochs}")
                t.set_postfix_str(f"Val Rest Loss: {val_rest_loss:.4f},  Val Mix Loss: {val_mix_loss:.4f}", refresh=True)
        

        # calculate and print metrics

        if plot == True:
            self._plot_losses()



    def _plot_losses(self):
        epochs = range(1, len(self.train_losses) + 1)
        train_rest_losses, train_mix_losses = zip(*self.train_losses)
        val_rest_losses, val_mix_losses = zip(*self.val_losses)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_rest_losses, label='Train Rest Loss')
        plt.plot(epochs, val_rest_losses, label='Val Rest Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Restoration Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mix_losses, label='Train Mix Loss')
        plt.plot(epochs, val_mix_losses, label='Val Mix Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Mixing Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def _test(self):
        print("testing")
        pass
