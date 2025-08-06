import numpy as np

class PolynomialKernelFeatureMap:
    """
    Implements the “reversed” polynomial‐kernel feature map:
        x = ((B @ c + 3) ** degree) / (3.5**degree * sqrt(p)) * ε,
    where
      - c ∈ ℝ^m is the input cost vector,
      - B ∈ {0,1}^{p×m} is a random Bernoulli mask,
      - ε ∼ Uniform(1−ε_bar, 1+ε_bar) is entrywise multiplicative noise.
    """

    def __init__(self, num_costs, num_features, *, degree=3, mask_prob=0.5, epsilon_bar=0.01, random_state=None, **kwargs):
        """
        Initialize the PolynomialKernelFeatureMap.

        Parameters
        ----------
        num_costs : int
            Dimensionality of the cost vector.
        num_features : int
            Number of output features.
        degree : int, Optional
            Polynomial degree. Default is 3.
        mask_prob : float, Optional
            Probability for B[i,j]=1 in the Bernoulli mask. Default is 0.5.
        epsilon_bar : float, Optional
            Half‐width of the uniform noise multiplier (ε ∈ [1−ε_bar,1+ε_bar]). Default is 0.01.
        random_state : int or None, Optional
            Seed for reproducibility. Default is None.
        **kwargs : dict, Optional
            Additional keyword arguments (not used in this implementation).
        """

        # Store required parameters
        self.num_costs = num_costs
        self.num_features = num_features

        # Store optional parameters
        self.degree = degree
        self.mask_prob = mask_prob
        self.epsilon_bar = epsilon_bar
        self.rng = np.random.default_rng(random_state)

        # draw the random Bernoulli mask once
        self.B = self.rng.binomial(1, self.mask_prob, size=(self.num_features, self.num_costs))

    def transform(self, c):
        """
        Generate the polynomial-kernel features for one or more cost vectors.

        Parameters
        ----------
        c : array-like of shape (n_samples, m) or (m,)
            Input cost matrix or a single cost vector. The second dimension
            must equal ``num_costs``.

        Returns
        -------
        ndarray of shape (n_samples, p) or (p,)
            The computed feature matrix (or vector for a single sample).
        """

        c = np.asarray(c, dtype=float)

        # Allow passing a single cost vector; treat it as one sample.
        was_1d = c.ndim == 1
        if was_1d:
            c = c[np.newaxis, :]

        if c.shape[1] != self.num_costs:
            raise ValueError(
                f"Expected input with {self.num_costs} costs, got {c.shape}"
            )

        n_samples = c.shape[0]

        # 1) linear projection + shift
        proj = c @ self.B.T               # shape (n_samples, p)
        shifted = proj + 3.0              # entrywise +3

        # 2) polynomial expansion and normalization
        base = shifted ** self.degree
        norm = (3.5 ** self.degree) * np.sqrt(self.num_features)
        base /= norm                      # shape (n_samples, p)

        # 3) multiplicative uniform noise in [1-ε_bar, 1+ε_bar]
        eps = self.rng.uniform(
            1 - self.epsilon_bar, 1 + self.epsilon_bar, size=(n_samples, self.num_features)
        )
        x = base * eps

        return x.squeeze() if was_1d else x
