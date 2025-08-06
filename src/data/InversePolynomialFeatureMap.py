import numpy as np


class InversePolynomialFeatureMap:
    """Generate features via an inverse polynomial style mapping.

    The transformation is given by

        x = (3.5 * (c / ε - 1) ** (1 / degree) - 3) * sqrt(p) * B,

    where

    * ``c`` ∈ ℝ^{n×m} is the input cost matrix,
    * ``ε`` has entries sampled i.i.d. from ``Uniform(1-ε_bar, 1+ε_bar)``,
    * ``B`` ∈ {0,1}^{m×p} is a Bernoulli mask with success probability
      ``mask_prob``, drawn once at initialization,
    * ``p`` is the number of output features.

    This mapping is intended as a counterpart to
    :class:`PolynomialKernelFeatureMap` but applies an inverse polynomial
    transformation followed by a random linear projection.
    """

    def __init__(
        self,
        num_costs,
        num_features,
        *,
        degree=3,
        mask_prob=0.5,
        epsilon_bar=0.01,
        random_state=None,
        **kwargs,
    ):
        """Initialize the feature map.

        Parameters
        ----------
        num_costs : int
            Dimensionality of the input cost vector ``c``.
        num_features : int
            Number of output features ``p``.
        degree : int, optional
            Degree ``d`` of the inverse polynomial transform. Default is 3.
        mask_prob : float, optional
            Probability for each entry of ``B`` to be one. Default is 0.5.
        epsilon_bar : float, optional
            Half-width of the uniform noise interval for ``ε``. Default is 0.01.
        random_state : int or None, optional
            Seed for reproducibility. Default is ``None``.
        **kwargs : dict, optional
            Additional keyword arguments (not used).
        """

        self.num_costs = num_costs
        self.num_features = num_features

        self.degree = degree
        self.mask_prob = mask_prob
        self.epsilon_bar = epsilon_bar
        self.rng = np.random.default_rng(random_state)

        # Bernoulli mask of shape (m, p)
        self.B = self.rng.binomial(1, self.mask_prob, size=(self.num_costs, self.num_features))

    def transform(self, c):
        """Compute features for one or more cost vectors.

        Parameters
        ----------
        c : array-like of shape (n_samples, m) or (m,)
            Input cost matrix or a single cost vector.

        Returns
        -------
        ndarray of shape (n_samples, p) or (p,)
            The computed feature matrix (or feature vector for a single sample).
        """

        c = np.asarray(c, dtype=float)

        was_1d = c.ndim == 1
        if was_1d:
            c = c[np.newaxis, :]

        if c.shape[1] != self.num_costs:
            raise ValueError(f"Expected input with {self.num_costs} costs, got {c.shape}")

        n_samples = c.shape[0]

        # Sample ε from Uniform(1-ε_bar, 1+ε_bar)
        eps = self.rng.uniform(1 - self.epsilon_bar, 1 + self.epsilon_bar, size=(n_samples, self.num_costs))

        # Inverse polynomial transform and scaling
        base = 3.5 * np.power(c / eps - 1.0, 1.0 / self.degree) - 3.0
        base *= np.sqrt(self.num_features)

        # Random projection using the Bernoulli mask
        x = base @ self.B

        return x.squeeze() if was_1d else x

