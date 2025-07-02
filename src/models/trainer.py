from typing import Tuple
from numpy import ndarray
import pyepo.metric
import torch
from torch.utils.data import DataLoader

class Trainer:
    """
    A class to handle the training and evaluation of a PyTorch model.
    """

    device: torch.device
    pred_model: torch.nn.Module
    opt_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_criterion: torch.nn.Module

    def __init__(self,
                 pred_model: torch.nn.Module,
                 opt_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 ) -> None:
        """
        Initializes the Trainer class.

        ------------
        Parameters
        ------------
        pred_model : torch.nn.Module
            The predictive model before the opt-layer to be trained.
        opt_model : torch.nn.Module
            The optimization model used as opt-layer.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training the model.
        loss_fn : torch.nn.Module
            The loss function to be used for training the model.
        device : torch.device, optional
            The device on which the model will be trained (default is 'cuda' if available,
            otherwise 'cpu').
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_model = pred_model.to(self.device)
        self.opt_model = opt_model
        self.optimizer = optimizer
        self.loss_criterion = loss_fn

    def train_epoch(self,
                    loader: DataLoader,
                    # loss_func: torch.nn.Module,
                    # optimizer: torch.optim.Optimizer
                    ) -> float:
        """
        Trains the model for one epoch.

        ------------
        Parameters
        ------------
        loader : DataLoader
            The DataLoader providing the training data.

        ------------
        Returns
        ------------
        float
            The average loss for the epoch. This is calculated as the total loss
            divided by the number of samples in the dataset.
        """

        running_loss = 0.0
        for feats, costs, sols, objs  in loader:

            # Move to GPU if specified
            feats = feats.to(self.device)
            costs = costs.to(self.device)
            sols = sols.to(self.device)
            objs = objs.to(self.device)

            

            # # Reset gradients
            # self.model.train()
            # self.optimizer.zero_grad()

            # Forward pass
            costs_pred = self.pred_model(feats)
            loss = self.loss_criterion(costs_pred, costs, sols, objs)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item() * feats.size(0)

        return running_loss / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the validation or test data.

        ------------
        Parameters
        ------------
        loader : DataLoader
            The DataLoader providing the validation or test data.
        
        ------------
        Returns
        ------------
        float
            The average loss for the evaluation. This is calculated as the total loss
            divided by the number of samples in the dataset.
        """

        self.pred_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for feats, costs, sols, objs in loader:
                # Move to GPU if specified
                feats = feats.to(self.device)
                costs = costs.to(self.device)
                sols = sols.to(self.device)
                objs = objs.to(self.device)

                # Forward pass
                costs_pred = self.pred_model(feats)

                # Compute loss
                total_loss += self.loss_criterion(costs_pred, costs, sols, objs).item() * feats.size(0)

        # Compute regret
        regret = pyepo.metric.regret(self.pred_model, self.opt_model, loader)

        return total_loss / len(loader.dataset), regret

    def fit(self,
            train_loader: DataLoader,
            test_loader: DataLoader = None,
            epochs: int = 10,
            n_epochs: int = -1
            ) -> ndarray[float]:
        """
        Fits the model to the training data.

        ------------
        Parameters
        ------------
        train_loader : DataLoader
            The DataLoader providing the training data.
        val_loader : DataLoader, optional
            The DataLoader providing the test data. 
            If not provided, no testing is performed.
        epochs : int, optional
            The number of epochs to train the model (default is 10).
        """

        # Set n_epochs so that the loss is printed no more than 10 times if not provided
        if n_epochs < 0:
            self.n_epochs = max(1, epochs // 10)

        # Initialize loss and regret vectors
        train_loss_vector = []
        train_regret_vector = []
        if test_loader is not None:
            test_loss_vector = []
            test_regret_vector = []

        # Training loop
        for epoch in range(epochs):
            # Train the model for one epoch
            train_loss = self.train_epoch(train_loader)

            # Evaluate training regret
            train_regret = pyepo.metric.regret(self.pred_model, self.opt_model, train_loader)

            # Append loss and regret to vectors
            train_loss_vector.append(train_loss)
            train_regret_vector.append(train_regret)

            # Print loss every n_epochs
            if epoch % self.n_epochs == 0:
                if test_loader:
                    test_loss, test_regret = self.evaluate(test_loader)
                    test_loss_vector.append(test_loss)
                    test_regret_vector.append(test_regret)
                    print(f"Epoch {epoch:02d} "
                          f"| Train Loss: {train_loss:.4f} "
                          f"| Train Regret: {train_regret:.4f} "
                          f"| Test Loss: {test_loss:.4f} "
                          f"| Test Regret: {test_regret:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:02d} "
                          f"| Train Loss: {train_loss:.4f} "
                          f"| Train Regret: {train_regret:.4f}"
                    )

        return (train_loss_vector, 
            train_regret_vector, 
            (test_loss_vector if test_loader else None), 
            (test_regret_vector if test_loader else None))
