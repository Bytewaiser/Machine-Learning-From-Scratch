import numpy as np

class OLS:
    """
        Ordinary least squares LinearRegression.
    """
    def fit(self, x, y):
        assert isinstance(x, (list, tuple, np.ndarray)), "x must be an array like."
        assert isinstance(y, (list, tuple, np.ndarray)), "y must be an array like."

        assert isinstance(x, np.ndarray) and x.ndim == 2, "x must be an 1d array."
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)

        self.x_std = np.std(x)
        self.y_std = np.std(y)

        self.r = np.corrcoef(x, y)[0,1]

    def predict(self, x):
        assert isinstance(x, (list, tuple, np.ndarray)), "x must be an array like."
        assert isinstance(x, np.ndarray) and x.ndim == 2, "x must be an 1d array."

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        m = self.r * self.y_std / self.x_std
        b = self.y_mean - m * self.x_mean
        return m*x + b

class LinearRegression:
    """
        LinearRegression using Linear Algebra.
    """
    def fit(self, x, y):
        assert isinstance(x, (list, tuple, np.ndarray)), "x must be an array like."
        assert isinstance(y, (list, tuple, np.ndarray)), "y must be an array like."

        A = np.insert(x, 0, 1, axis=1)
        Ax = np.dot(A.T, A)
        b = np.dot(A.T, y)
        self.w = np.linalg.solve(Ax, b)

    def predict(self, x):
        assert isinstance(x, (list, tuple, np.ndarray)), "x must be an array like."
        x = np.insert(x, 0, 1, axis=1)
        return np.dot(x, self.w)

