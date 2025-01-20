# train.py
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
