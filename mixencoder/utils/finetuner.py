import torch.nn as nn
import torch
import mixencoder.models
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class FineTuner:
    """Finetuner for MixEncoder.
    Args: model (MixEncoder): trained model with an encoder_stack.
    """
    def __init__(self, model):
        super(FineTuner, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.metric = None
        self.loss_metric = None
        self.model = model



    def finetune(self, 
            x_train: torch.Tensor, 
            y_train: torch.Tensor, 
            x_val: torch.Tensor, 
            y_val: torch.Tensor, 
            plot: bool = False, 
            epochs: int = 100, 
            mode: str = "mse", 
            metric: str = "accuracy",
            batch_size: int = 200,
            lr: float = 1e-3):
        
        """Fit the model on the dataset.
        Args:
            x_train (torch.Tensor): Training data
            y_train (torch.Tensor): Training labels
            x_val (torch.Tensor): Validation data
            y_val (torch.Tensor): Validation labels
            epochs (int): Number of epochs
            mode (str): Mode of the model - reg, cls, mcls (regression, binary classification, multi-class classification)
            metric (str): Metric to evaluate - mse, accuracy, AUC
            batch_size (int): Batch size
            lr (float): Learning rate
        """
        
        assert mode in ["reg", "cls", "mcls"], "Invalid mode - [reg, cls, mcls]"
        assert metric in ["accuracy", "AUC", "MSE"], "Invalid metric - [accuracy, AUC, MSE]"
        if mode == "reg":
            assert metric == "MSE", "For regression mode, metric must be MSE"
        else:
            assert metric != "MSE", "For classification modes, metric cannot be MSE"
        assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same number of samples"
        assert x_val.shape[0] == y_val.shape[0], "x_val and y_val must have the same number of samples"
        assert isinstance(x_train, torch.Tensor), "x_train must be a torch.Tensor"
        assert isinstance(y_train, torch.Tensor), "y_train must be a torch.Tensor"
        assert isinstance(x_val, torch.Tensor), "x_val must be a torch.Tensor"
        assert isinstance(y_val, torch.Tensor), "y_val must be a torch.Tensor"


        if mode == "reg":
            criterion = nn.MSELoss()
            self.loss_metric = "MSE"
        elif mode == "cls":
            criterion = nn.BCELoss() 
            self.loss_metric = "BCE" 
        else:
            criterion = nn.CrossEntropyLoss()
            self.loss_metric = "MCE"

        self.metric = metric

        # define data loaders
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

        # define prediction head
        prediction_head = mixencoder.models.PredictionHead(input_size=self.model.emb_size, mode=mode)
        prediction_head.to(self.model.device)

        # define optimizers
        optimizer = torch.optim.Adam(self.model.encoder_stack.parameters(), lr=lr)
        pred_optimizer = torch.optim.Adam(prediction_head.parameters(), lr=lr)

        t = tqdm.tqdm(range(epochs))

        for epoch in t:
            self.model.encoder_stack.train()
            prediction_head.train()
            train_loss = 0
            train_metric = 0
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                z = self.model.encoder_stack(x)
                pred = prediction_head(z)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                pred_optimizer.step()
                train_loss += loss.item()
                train_metric += self._calculate_metric(metric, y, pred)
            train_loss /= len(train_loader)
            train_metric /= len(train_loader)
            self.train_losses.append((train_loss))
            self.train_metrics.append((train_metric))
            t.set_description(f"Finetune Train Epoch {epoch+1}/{epochs}")
            t.set_postfix_str(f"Finetuned Train {metric}: {train_metric}, Finetuned Train Loss: {train_loss}")

            self.model.encoder_stack.eval()
            prediction_head.eval()
            val_loss = 0
            val_metric = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    z = self.model.encoder_stack(x)
                    pred = prediction_head(z)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
                    val_metric += self._calculate_metric(metric, y, pred)
                val_loss /= len(val_loader)
                val_metric /= len(val_loader)
                self.val_losses.append((val_loss))
                self.val_metrics.append((val_metric))
                t.set_description(f"Finetune Val Epoch {epoch+1}/{epochs}")
                t.set_postfix_str(f"Finetuned Val {metric}: {val_metric:.4f}, Finetuned Val Loss: {val_loss:.4f}")
        
        if plot:
            self._plot_losses()



    def _calculate_metric(self, metric, y_true, y_pred):
        """Calculate metric."""
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        if metric == "accuracy":
            y_pred = (y_pred > 0.5).astype(int)
            return (y_true == y_pred).mean()
        elif metric == "AUC":
            return roc_auc_score(y_true, y_pred)
        elif metric == "MSE":
            return ((y_true - y_pred) ** 2).mean()
        else:
            raise ValueError("Invalid metric - [accuracy, AUC, MSE]")
        

    
    def _plot_losses(self):
        """Plot losses."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.title(f"Finetuning Losses ({self.loss_metric})")

        plt.subplot(1, 2, 2)
        plt.plot(self.train_metrics, label=f"Train {self.metric}")
        plt.plot(self.val_metrics, label=f"Val {self.metric}")
        plt.xlabel("Epochs")
        plt.ylabel(self.metric)
        plt.legend()
        plt.grid()
        plt.title(f"Finetuning Results - {self.metric}")
        
        plt.tight_layout()
        plt.show()

        print(f"Highest Validation {self.metric}: {max(self.val_metrics)}\nFinal Validation {self.metric}: {self.val_metrics[-1]}")