from copy import deepcopy
import numpy as np

from src.data.PolynomialKernelFeatureMap import PolynomialKernelFeatureMap
from src.data.InversePolynomialFeatureMap import InversePolynomialFeatureMap
from src.data.config import HP

class DataGenerator:

    def __init__(self, 
                 num_costs,
                 num_features, 
                 cost_feature_map: str = "PolynomialKernel", 
                 c_range: tuple = (0, 10),
                 **kwargs):
        """
        Initialize the DataGenerator with a cost feature map.

        Parameters:
        -----------
        """
        
        # Initialize the cost feature map
        random_seed = kwargs.pop("random_state", HP.random_seed)
        if cost_feature_map == "PolynomialKernel":
            self.cost_feature_map = PolynomialKernelFeatureMap(
                num_costs, num_features, random_state=random_seed, **kwargs
            )
        elif cost_feature_map == "InversePolynomial":
            self.cost_feature_map = InversePolynomialFeatureMap(
                num_costs, num_features, random_state=random_seed, **kwargs
            )
        else:
            raise ValueError(f"Unknown cost feature map: {cost_feature_map}")

        # Store the number of data samples and features
        self.num_costs = num_costs
        self.num_features = num_features
        self.cost_range = c_range

    def __deepcopy__(self, memo):
        """
        Create a deepcopy of the DataGenerator instance.

        Parameters:
        -----------
        memo : dict
            A memo dictionary to keep track of already copied objects.

        Returns:
        --------
        DataGenerator
            A new instance of DataGenerator with the same cost feature map.
        """
        return DataGenerator(deepcopy(self.cost_feature_map, memo))

    def generate_data(self, num_samples):
        """
        Generate a dataset of cost vectors and their corresponding features.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.
        cost_range : tuple
            The range (min, max) for the cost values.

        Returns:
        --------
        tuple
            A tuple containing an array of cost vectors and their corresponding feature vectors.
        """

        # Generate random costs within the specified range
        costs = np.random.uniform(self.cost_range[0], self.cost_range[1], (num_samples, self.num_costs))

        # Generate features using the cost feature map
        # Note that the costs are normalized to the range [0, 1] first
        features = self.cost_feature_map.transform(costs/(self.cost_range[1] - self.cost_range[0]) + self.cost_range[0])

        return costs, features
    
    def generate_features(self, costs):
        """
        Generate features for a given set of cost vectors.

        Parameters:
        -----------
        costs : array-like
            An array of cost vectors.

        Returns:
        --------
        ndarray
            An array of feature vectors corresponding to the input costs.
        """

        return self.cost_feature_map.transform(costs)
    

