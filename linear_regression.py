import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, input_dim: int, lr: float = 1e-3):
        self.input_dim = input_dim
        self.W = np.random.randn(input_dim + 1) / np.sqrt(input_dim)  # +1 for bias term
        self.lr = lr
    
    def _validate_input(self, X):
        # X shape => (N, input_dims)
        if len(X.shape) != 2:
            raise ValueError(f"len(X.shape) should be 2 not {len(X.shape)}")
        
        if X.shape[1] != self.input_dim:
            raise ValueError("X.shape[1] should be equal to input_dim "
                f"({self.input_dim}) not {X.shape[1]}")
    
    def _augment_input(self, X):
        self._validate_input(X)
        # add column of 1's to X, so shape should be (N, input_dims + 1)
        return np.hstack((X, np.ones((X.shape[0], 1))))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._augment_input(X) @ self.W  # shape => (N, )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # X shape => (N, input_dims)
        # y shape => (N, )
        X = self._augment_input(X)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X.shape[0] ({X.shape[0]}) must equal y.shape[0] ({y.shape[0]})")
        if len(y.shape) != 1:
            raise ValueError(f"len(y.shape) should be 1 not {len(y.shape)}")
        
        grad = 2 * (-X.T @ y + X.T @ X @ self.W)
        self.W -= self.lr * grad
        return grad


class EligibilityTraceLR(LinearRegression):
    def __init__(self, gamma: float, lbda: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.lbda = lbda
        self.E = np.zeros(self.W.shape)
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # X shape => (N, input_dims)
        # y shape => (N, )
        X = self._augment_input(X)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X.shape[0] ({X.shape[0]}) must equal y.shape[0] ({y.shape[0]})")
        if len(y.shape) != 1:
            raise ValueError(f"len(y.shape) should be 1 not {len(y.shape)}")
        
        grad = 2 * (-X.T @ y + X.T @ X @ self.W)
        self.E = grad + self.gamma * self.lbda * self.E
        self.W -= self.lr * self.E
        return grad

if __name__ == "__main__":
    lin_reg = LinearRegression(1)
    w, b = 2, 1
    x = np.linspace(-10, 10, 21)
    y = w*x + b

    for epoch in range(1000):
        lin_reg.partial_fit(x[None, :].T, y)
    
    assert np.isclose([w, b], lin_reg.W).all()

        


        
