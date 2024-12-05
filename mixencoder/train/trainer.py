# Training logic for training mixencoder.
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """Trainer class for training a model with an associated setup."""
    def __init__(self):
        super(Trainer, self).__init__()

        self.device = "cpu"
        self.train_losses = []
        self.val_losses = []


    def fit(self, 
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_val: torch.Tensor,
            y_val: torch.Tensor,
            epochs: int = 100,
            batch_size: int = 200,
            lr: float = 1e-3,
            plot: bool = True,
            device: str = "cpu",
            optimizer: str = "adam"):
        
        # TODO: add testing functionality
        
        """Training loop for a model with an associated setup.
        Args:
            x_train (torch.Tensor): Training data
            y_train (torch.Tensor): Training labels
            x_val (torch.Tensor): Validation data
            y_val (torch.Tensor): Validation labels
            train_data (torch.Tensor): Training data
            val_data (torch.Tensor): Validation data
            epochs (int): Number of epochs
            batch_size (int): Batch size
            lr (float): Learning rate
            plot (bool): Plot losses
            device (str): Device to train on - cpu, cuda
        """

        
        # assertions for input validation

        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'"
        assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same number of samples"
        assert x_val.shape[0] == y_val.shape[0], "x_val and y_val must have the same number of samples"
        assert optimizer in ["adam", "sgd"], "Optimizer must be either 'adam' or 'sgd'"
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


        optimizers = {
            "adam" : optim.Adam,
            "sgd" : optim.SGD
        }

        # set training parameters
        # TODO: modularize this so it can be reused in finetuner

        setup = self._setup()

        self.to(self.device)
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)
        optimizer = optimizers[optimizer](self.parameters(), lr=lr)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        self.train_losses, self.val_losses = [], []


        # main training and validation loop

        t = tqdm.tqdm(range(epochs), nrows=3)
        v = tqdm.tqdm(range(epochs), nrows=3)

        for epoch in t:
            self._training_and_validation_step(setup, t, v, epochs, epoch, train_loader, val_loader, optimizer)
      

        # calculate and print metrics

        if plot:
            self._plot_losses(setup["losses"], epochs=epochs)


    def _training_and_validation_step(self, setup, t, v, epochs, epoch, train_loader, val_loader, optimizer):
        for i, ((x_train, y_train), (x_val, y_val)) in enumerate(zip(train_loader, val_loader)):
            train_loss = self._training_step(x_train, y_train, t, setup, epochs, epoch, optimizer)
            train_loss = [l/len(train_loader) for l in train_loss]
            self.train_losses.append(tuple(train_loss))
            
            val_loss = self._validation_step(x_val, y_val, v, setup, epochs, epoch)
            val_loss = [l/len(val_loader) for l in val_loss]
            self.val_losses.append(tuple(val_loss))


    def _training_step(self, x, y, t, setup, epochs, epoch, optimizer):
        self.train()
        loss = [0 for _ in range(len(setup["losses"]))]
        grad_loss = 0
        out = self(x)
        tstr = ""
        for j, ls in enumerate(setup["losses"]):
            optimizer.zero_grad()
            grad_loss += setup["losses"][ls]["type"](out[setup["losses"][ls]["pred"]], out[setup["losses"][ls]["target"]])*setup["losses"][ls]["weight"]
            loss[j] += grad_loss.item()
            tstr += f"{ls}: {loss[j]:.4f} "
        grad_loss.backward()
        optimizer.step()
        t.set_description(f"Train Epoch {epoch+1}/{epochs}")
        t.set_postfix_str(tstr, refresh=True)
        return loss


    def _validation_step(self, x, y, v, setup, epochs, epoch):
        self.eval()
        loss = [0 for _ in range(len(setup["losses"]))]
        tstr = ""
        grad_loss = 0
        with torch.no_grad():
            out = self(x)
            for j, ls in enumerate(setup["losses"]):
                grad_loss += setup["losses"][ls]["type"](out[setup["losses"][ls]["pred"]], out[setup["losses"][ls]["target"]])*setup["losses"][ls]["weight"]
                loss[j] += grad_loss.item()
                tstr += f"Val {ls}: {loss[j]:.4f} "
        v.set_description(f"Val Epoch {epoch+1}/{epochs}")
        v.set_postfix_str(tstr, refresh=True)
        return loss


    def _plot_losses(self, losses, epochs):
        import numpy as np
        x = np.linspace(1, epochs, len(self.train_losses))

        num_losses = len(losses)
        fig, axes = plt.subplots(1, num_losses, figsize=(15, 5), constrained_layout=True)

        if num_losses == 1:
            axes = [axes]

        # Plot data
        for i, ls in enumerate(losses):
            ax = axes[i]
            train = [t[i] for t in self.train_losses]
            val = [v[i] for v in self.val_losses]
            ax.plot(x, train, label=f'Train {ls}')
            ax.plot(x, val, label=f'Validation {ls}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(ls)
            ax.grid()
            ax.legend()

        plt.show()
       
    def _test(self):
        print("testing")

