from typing import Optional, Tuple
from unicodedata import name
from matplotlib.axes import Axes
from numpy import ndarray
from numpy import arange
import pyepo.metric
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dflintdpy.data.config import HP

class DFLTrainer:
    """
    A class to handle the training and evaluation of a PyTorch model.
    """

    device: torch.device
    pred_model: torch.nn.Module
    opt_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_criterion: torch.nn.Module
    method_name: str

    def __init__(self,
                 pred_model: torch.nn.Module,
                 opt_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 method_name: str = "spo+",
                 cfg: HP = None,
                 aggregate: str = "mean",
                 cvar_alpha: float = 0.9,
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
        aggregate : str, optional
            How to aggregate scenario losses for each base instance. Options
            are ``"mean"`` (default), ``"worst"`` or ``"cvar"``.
        cvar_alpha : float, optional
            Confidence level used when ``aggregate='cvar'``. The trainer
            averages the worst ``(1 - alpha)`` fraction of scenario losses.
        device : torch.device, optional
            The device on which the model will be trained (default is 'cuda' if available,
            otherwise 'cpu').
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_model = pred_model.to(self.device)
        self.opt_model = opt_model
        self.optimizer = optimizer
        self.loss_criterion = loss_fn
        if method_name in type(self).VALID_METHODS():
            self.method_name = method_name
        else:
            raise ValueError(f"Unknown method name: {method_name}\n"
                             f"Valid methods are: {type(self).VALID_METHODS}")
        self.cfg = cfg
        if cfg is None and method_name == "hybrid":
            raise ValueError("Configuration must be provided for hybrid method.")

        self.aggregate = aggregate
        self.cvar_alpha = cvar_alpha

    def train_epoch(self,
                    loader: DataLoader
                    ) -> float:
        """
        Trains the model for one epoch.

        ------------
        Parameters
        ------------
        loader : DataLoader
            The DataLoader providing the training data. Each batch should
            return ``feats`` with shape ``(B, p)`` and ``costs``, ``sols``,
            ``objs`` and ``intds`` with shape ``(B, K, ...)`` where ``K`` is
            the number of scenarios per base instance.

        ------------
        Returns
        ------------
        float
            The average loss for the epoch. This is calculated as the total loss
            divided by the number of samples in the dataset.
        """

        self.pred_model.train()
        running_loss = 0.0
        for feats, costs, sols, objs, intds in loader:

            feats = feats.to(self.device)
            costs = costs.to(self.device)
            sols = sols.to(self.device)
            objs = objs.to(self.device)
            intds = intds.to(self.device)

            pred = self.pred_model(feats).unsqueeze(1) + intds
            B, K = costs.shape[:2]

            loss_flat = type(self).compute_loss(
                self.loss_criterion,
                pred.view(B * K, *pred.shape[2:]),
                costs.view(B * K, *costs.shape[2:]),
                sols.view(B * K, *sols.shape[2:]),
                objs.view(B * K, *objs.shape[2:]),
                method_name=self.method_name,
            )

            if loss_flat.dim() == 0:
                loss_per_scen = loss_flat.repeat(B, K)
            else:
                loss_per_scen = loss_flat.view(B, K, -1).mean(-1)
            if self.aggregate == "mean":
                loss = loss_per_scen.mean(dim=1).mean()
            elif self.aggregate == "worst":
                loss = loss_per_scen.max(dim=1).values.mean()
            elif self.aggregate == "cvar":
                k_tail = max(1, min(K, int((1 - self.cvar_alpha) * K)))
                topk = loss_per_scen.topk(k_tail, dim=1).values
                loss = topk.mean(dim=1).mean()
            else:
                raise ValueError(f"Unknown aggregate: {self.aggregate}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * B

        return running_loss / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the validation or test data.

        ------------
        Parameters
        ------------
        loader : DataLoader
            The DataLoader providing the validation or test data. Batches have
            the same shape convention as in :func:`train_epoch`.
        
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
            for feats, costs, sols, objs, intds in loader:
                feats = feats.to(self.device)
                costs = costs.to(self.device)
                sols = sols.to(self.device)
                objs = objs.to(self.device)
                intds = intds.to(self.device)

                pred = self.pred_model(feats).unsqueeze(1) + intds
                B, K = costs.shape[:2]

                loss_flat = type(self).compute_loss(
                    self.loss_criterion,
                    pred.view(B * K, *pred.shape[2:]),
                    costs.view(B * K, *costs.shape[2:]),
                    sols.view(B * K, *sols.shape[2:]),
                    objs.view(B * K, *objs.shape[2:]),
                    self.method_name,
                )

                if loss_flat.dim() == 0:
                    loss_per_scen = loss_flat.repeat(B, K)
                else:
                    loss_per_scen = loss_flat.view(B, K, -1).mean(-1)
                if self.aggregate == "mean":
                    loss = loss_per_scen.mean(dim=1).mean()
                elif self.aggregate == "worst":
                    loss = loss_per_scen.max(dim=1).values.mean()
                elif self.aggregate == "cvar":
                    k_tail = max(1, min(K, int((1 - self.cvar_alpha) * K)))
                    topk = loss_per_scen.topk(k_tail, dim=1).values
                    loss = topk.mean(dim=1).mean()
                else:
                    raise ValueError(f"Unknown aggregate: {self.aggregate}")

                total_loss += loss.item() * B
        # Compute regret
        loader.normal_mode() # evaluate regret only on original samples
        regret = pyepo.metric.regret(self.pred_model, self.opt_model, loader)
        loader.adverse_mode() # reset to adverse mode

        return total_loss / len(loader.dataset), regret

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
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
        n_epochs : int, optional
            The frequency of printing the loss during training.
            If set to -1, it will be set to max(1, epochs // 10) to print the loss no more than 10 times.
            If set to a positive integer, it will print the loss every n_epochs epochs.
        """

        # Set data loaders to adverse mode
        train_loader.adverse_mode()
        if val_loader is not None:
            val_loader.adverse_mode()

        # Set n_epochs so that the loss is printed no more than 10 times if not provided
        if n_epochs < 0:
            self.n_epochs = max(1, epochs // 10)

        # Initialize train loss and regret vectors
        train_loss, train_regret = self.evaluate(train_loader)
        train_loss_vector = [train_loss]
        train_regret_vector = [train_regret]

        # If test_loader is provided, initialize test loss and regret vectors
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
            # Set lambda for hybrid method
            if self.method_name == "hybrid":
                self.loss_criterion.lam = DFLTrainer.lambda_schedule(self.cfg, epoch)

            # Train the model for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate training regret
            train_loader.normal_mode()  # evaluate regret only on original samples
            train_regret = pyepo.metric.regret(self.pred_model, 
                                               self.opt_model, 
                                               train_loader)
            train_loader.adverse_mode()  # reset to adverse mode

            # Append loss and regret to vectors
            train_loss_vector.append(train_loss)
            train_regret_vector.append(train_regret)
            
            # Print loss every n_epochs
            if (epoch + 1) % self.n_epochs == 0:
                if val_loader:
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
    def VALID_METHODS():
        """
        Returns the list of valid method names for training.
        """
        return ["spo+", "hybrid", "ptb", "pfy", "imle", "aimle", "nce", "cmap",
                "dbb", "nid", "pg", "ltr"]

    @staticmethod
    def compute_loss(loss_criterion: torch.nn.Module,
                    costs_pred: torch.Tensor,
                    costs: torch.Tensor,
                    sols: torch.Tensor,
                    objs: torch.Tensor,
                    method_name: str) -> torch.Tensor:
        """
        Computes the loss for the given method name.

        ------------
        Parameters
        ------------
        loss_criterion : torch.nn.Module
            The loss criterion to use for computing the loss.
        costs_pred : torch.Tensor
            The predicted costs.
        costs : torch.Tensor
            The true costs.
        sols : torch.Tensor
            The solutions.
        objs : torch.Tensor
            The objectives.
        method_name : str
            The name of the method to use for computing the loss.

        ------------
        Returns
        ------------
        torch.Tensor
            The computed loss.
        """

        if method_name == "spo+":
            try:
                return loss_criterion(costs_pred, costs, sols, objs, reduction="none")
            except TypeError:
                return loss_criterion(costs_pred, costs, sols, objs)
        if method_name == "hybrid":
            try:
                return loss_criterion(costs_pred, costs, sols, objs, reduction="none")
            except TypeError:
                return loss_criterion(costs_pred, costs, sols, objs)
        elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
            try:
                return loss_criterion(costs_pred, sols, reduction="none")
            except TypeError:
                return loss_criterion(costs_pred, sols)
        elif method_name in ["dbb", "nid"]:
            try:
                return loss_criterion(costs_pred, costs, objs, reduction="none")
            except TypeError:
                return loss_criterion(costs_pred, costs, objs)
        elif method_name in ["pg", "ltr"]:
            try:
                return loss_criterion(costs_pred, costs, reduction="none")
            except TypeError:
                return loss_criterion(costs_pred, costs)
    
    @staticmethod
    def lambda_schedule(cfg, epoch):
        # Example: warm start with strong anchor, then linear decay
        if epoch < cfg.get("spo_po_epochs"):
            return 1.0             # train without SPO for first epochs
        else:
            return cfg.get("lam")  # use constant lambda afterwards
        # if epoch < 33:
        #     return 0.7             # strong anchor for 3 epochs
        # elif epoch < 45:
        #     # decay to 0.1 by epoch 45
        #     t = (epoch - 33) / (45 - 33)
        #     return (1 - t) * 0.7 + t * 0.1
        # else:
        #     return 0.05            # long tail


    @staticmethod
    def vis_learning_curve(trainer: "DFLTrainer",
                        train_loss_log: ndarray[float],
                        train_regret_log: ndarray[float],
                        test_loss_log: ndarray[float] = None,
                        test_regret_log: ndarray[float] = None,
                        ax: Optional[Axes] = None,
                        file_name: Optional[str] = None) -> None:
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
        ax : Optional[Axes], optional
            The matplotlib Axes to plot on. If not provided, a new figure is created.
        file_name : Optional[str], optional
            The file name to save the plot. If not provided, the plot is not saved.
        ------------
        """

        # Create figure and subplots
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
        else:
            ax1, ax2 = ax

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
        ax1.set_yscale('log')
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
        # plt.show()
        if file_name is not None:
            plt.savefig(file_name + ".png")
            plt.close()
        pass
