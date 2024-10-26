import numpy as np

class LinearRegression:
    def __init__(self):
        self.m = 0  # slope
        self.b = 0  # intercept
    
    def fit(self, X, y):
        n = len(X)
        # Calculate the slope (m) and intercept (b)
        self.m = (n * np.dot(X, y) - np.sum(X) * np.sum(y)) / (n * np.dot(X, X) - np.sum(X) ** 2)
        self.b = (np.sum(y) - self.m * np.sum(X)) / n

    def predict(self, X):
        return self.m * X + self.b

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Predicted values:", predictions)
