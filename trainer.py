import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 ) -> None:
        """
        Initializes the Trainer class.

        ------------
        Parameters
        ------------
        model : torch.nn.Module
            The model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training the model.
        loss_fn : torch.nn.Module
            The loss function to be used for training the model.
        device : torch.device, optional
            The device on which the model will be trained (default is 'cuda' if available,
            otherwise 'cpu').
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_criterion = loss_fn

    def train_epoch(self, loader: DataLoader) -> float:
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
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(self.device)
            c_batch = c_batch.to(self.device)

            # Reset gradients
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            c_pred = self.model(x_batch)
            loss = self.loss_criterion(c_pred, c_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        return running_loss / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> float:
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

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, c_batch in loader:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                c_pred = self.model(x_batch)
                total_loss += self.loss_criterion(c_pred, c_batch).item() * x_batch.size(0)
        return total_loss / len(loader.dataset)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            epochs: int = 100,
            n_epochs: int = -1
            ) -> None:
        """
        Fits the model to the training data.

        ------------
        Parameters
        ------------
        train_loader : DataLoader
            The DataLoader providing the training data.
        val_loader : DataLoader, optional
            The DataLoader providing the validation data. If not provided, no validation will be performed.
        epochs : int, optional
            The number of epochs to train the model (default is 100).
        """

        # Set n_epochs so that the loss is printed no more than 10 times if not provided
        if n_epochs < 0:
            self.n_epochs = max(1, epochs // 10)

        # Initialize loss vectors
        train_loss_vector = []
        if val_loader:
            val_loss_vector = []

        # Training loop
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            train_loss_vector.append(train_loss)

            # Print loss every n_epochs
            if epoch % self.n_epochs == 0:
                if val_loader:
                    val_loss = self.evaluate(val_loader)
                    val_loss_vector.append(val_loss)
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")


