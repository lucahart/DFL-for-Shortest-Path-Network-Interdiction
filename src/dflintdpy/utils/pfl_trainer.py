import pyepo.metric
import torch

from typing import Tuple
from numpy import ndarray
from numpy import arange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dflintdpy.data.adverse.adverse_loader import AdvLoader

class PFLTrainer:
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
                 loss_fn: torch.nn.Module
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
                    loader: DataLoader
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

        self.pred_model.train()
        running_loss = 0.0
        # Dataset is expected to be in normal mode, which omits interdictions
        # so each batch contains only features, costs, solutions and objectives.
        for feats, costs, _, _ in loader:

            # Move to GPU if specified
            feats = feats.to(self.device)
            costs = costs.to(self.device)

            # Forward pass
            costs_pred = self.pred_model(feats)
            loss = self.loss_criterion(costs_pred, costs)

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
            # Dataset is expected to be in normal mode, which omits interdictions
            # so each batch contains only features, costs, solutions and objectives.
            for feats, costs, _, _ in loader:

                # Move to GPU if specified
                feats = feats.to(self.device)
                costs = costs.to(self.device)

                # Forward pass
                costs_pred = self.pred_model(feats)

                # Compute loss
                total_loss += self.loss_criterion(costs_pred, costs).item() * feats.size(0)
        # Compute regret
        regret = pyepo.metric.regret(self.pred_model, self.opt_model, loader)

        return total_loss / len(loader.dataset), regret

    def fit(self,
            train_loader: AdvLoader,
            val_loader: AdvLoader = None,
            epochs: int = 10,
            n_epochs: int = -1
            ) -> ndarray[float]:
        """
        Fits the model to the training data.

        ------------
        Parameters
        ------------
        train_loader : AdvLoader
            The AdvLoader providing the training data.
        val_loader : AdvLoader, optional
            The AdvLoader providing the test data. 
            If not provided, no testing is performed.
        epochs : int, optional
            The number of epochs to train the model (default is 10).
        n_epochs : int, optional
            The frequency of printing the loss during training.
            If set to -1, it will be set to max(1, epochs // 10) to print the loss no more than 10 times.
            If set to a positive integer, it will print the loss every n_epochs epochs.
        """

        # Set data loaders to normal mode
        train_loader.normal_mode()
        if val_loader is not None:
            val_loader.normal_mode()

        # Set n_epochs so that the loss is printed 
        # No more than 10 times if not provided
        if n_epochs < 0:
            self.n_epochs = max(1, epochs // 10)

        # Initialize train loss and regret vectors
        train_loss, train_regret = self.evaluate(train_loader)
        train_loss_vector = [train_loss]
        train_regret_vector = [train_regret]

        # If val_loader is provided, initialize val loss and regret vectors
        if val_loader is not None:
            test_loss, test_regret = self.evaluate(val_loader)
            test_loss_vector = [test_loss]
            test_regret_vector = [test_regret]
            
        # Print the initial evaluation before starting training
        if val_loader is not None:
            print(
                f"Epoch {0:02d} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Regret: {train_regret:.4f} "
                f"| Validation Loss: {test_loss:.4f} "
                f"| Validation Regret: {test_regret:.4f}"
            )
        else:
            print(
                f"Epoch {0:02d} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Regret: {train_regret:.4f}"
            )


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
            if (epoch + 1) % self.n_epochs == 0:
                if val_loader is not None:
                    test_loss, test_regret = self.evaluate(val_loader)
                    test_loss_vector.append(test_loss)
                    test_regret_vector.append(test_regret)
                    print(f"Epoch {epoch+1:02d} "
                          f"| Train Loss: {train_loss:.4f} "
                          f"| Train Regret: {train_regret:.4f} "
                          f"| Validation Loss: {test_loss:.4f} "
                          f"| Validation Regret: {test_regret:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1:02d} "
                          f"| Train Loss: {train_loss:.4f} "
                          f"| Train Regret: {train_regret:.4f}"
                    )

        return (train_loss_vector, 
            train_regret_vector, 
            (test_loss_vector if val_loader else None), 
            (test_regret_vector if val_loader else None))


    @staticmethod
    def vis_learning_curve(trainer: "PFLTrainer",
                        train_loss_log: ndarray[float],
                        train_regret_log: ndarray[float],
                        test_loss_log: ndarray[float] = None,
                        test_regret_log: ndarray[float] = None) -> None:
        """
        Visualizes the learning curve of the model during training.

        ------------
        Parameters
        ------------
        trainer : Trainer
            The Trainer instance containing the model and training parameters.
        train_loss_log : ndarray[float]
            The training loss log.
        train_regret_log : ndarray[float]
            The training regret log.
        test_loss_log : ndarray[float], optional
            The testing loss log. If not provided, no testing data is plotted.
        test_regret_log : ndarray[float], optional
            The testing regret log. If not provided, no testing data is plotted.
        ------------
        """

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

        # Plot regret learning curve with training and testing data
        ax1.plot(train_regret_log, marker='.', label='Training Regret')
        if test_regret_log is not None:
            ax1.scatter(
                arange(len(test_regret_log))*trainer.n_epochs, 
                test_regret_log, 
                marker='x', 
                color='red', 
                label='Testing Regret'
            )
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Regret')
        ax1.set_title('Regret Learning Curve')
        ax1.legend()
        

        # Plot loss learning curve with training and testing data
        ax2.plot(train_loss_log, marker='.', label='Training Loss')
        if test_loss_log is not None:
            ax2.scatter(arange(len(test_loss_log))*trainer.n_epochs, 
                test_loss_log, 
                marker='x', 
                color='red', 
                label='Testing Loss'
            )
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_yscale('log')
        ax2.set_title('Loss Learning Curve')
        ax2.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
        pass
