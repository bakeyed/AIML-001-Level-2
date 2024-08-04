import numpy as np
class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        Number of passes over the training set
    Attributes
    ----------
    w_ : array-like, shape = [n_features, 1]
        Weights after fitting the model
    cost_ : list
        Total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta  # Learning rate
        self.n_iterations = n_iterations  # Number of iterations

    def fit(self, x, y):
        """Fit the model to the training data.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
            Returns the instance itself
        """
        self.cost_ = []  # List to store cost values
        self.w_ = np.zeros((x.shape[1], 1))  # Initialize weights
        m = x.shape[0]  # Number of training examples

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)  # Predict values
            residuals = y_pred - y  # Compute residuals
            gradient_vector = np.dot(x.T, residuals)  # Calculate gradient
            self.w_ -= (self.eta / m) * gradient_vector  # Update weights
            cost = np.sum((residuals ** 2)) / (2 * m)  # Compute cost
            self.cost_.append(cost)  # Store cost
        return self

    def predict(self, x):
        """Predict values for new data after model training.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        array-like, shape = [n_samples, 1]
            Predicted values
        """
        return np.dot(x, self.w_)  # Return predictions

        mse = np.sum((y_pred - y_actual)**2) # mean square error
        rmse = np.sqrt(mse/m) #Root mean square error, m is no of training eg

        ssr = np.sum((y_pred - y_actual)**2) # sum of square of residuals
        sst = np.sum((y_actual - np.mean(y_actual))**2) #  total sum of squares
        r2_score = 1 - (ssr/sst) # R2 score