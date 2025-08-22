from torch import nn

class TwoLayerSigmoid(nn.Module):
    """Two-layer neural network with sigmoid activation."""

    fc1: nn.Linear
    fc2: nn.Linear
    act: nn.Sigmoid

    def __init__(self, num_feat: int, hidden_dim: int, num_edges: int) -> None:
        """Initializes the network.

        Parameters
        ----------
        num_feat: int
            Number of input features.
        hidden_dim: int
            Hidden layer width.
        num_edges: int
            Number of predicted edge weights.
        """
        super().__init__()
        self.fc1 = nn.Linear(num_feat, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_edges)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Forward pass of the network."""
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
