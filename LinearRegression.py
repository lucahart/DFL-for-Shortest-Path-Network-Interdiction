from torch import nn

# build linear model
class LinearRegression(nn.Module):

    def __init__(self, num_feat: int, num_edges: int) -> None:
        """
        Constructs a linear regression model for predicting a graph's edge weights.
        
        ----------
        Parameters
        ----------

        """

        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_edges)
        pass

    def forward(self, x):
        """
        Forward pass of the linear regression model.
        """

        out = self.linear(x)
        return out
    
    def backward(self, loss):
        """
        Backward pass of the linear regression model.
        """

        
        
        
