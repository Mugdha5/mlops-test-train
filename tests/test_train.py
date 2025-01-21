import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from train import model  # Import the trained model from train.py

def test_model_coefficients():
    """Test if the model coefficients are close to the expected value."""
    expected_coefficient = np.array([2])  # Since y = 2x
    np.testing.assert_array_almost_equal(model.coef_, expected_coefficient, decimal=5)

def test_model_intercept():
    """Test if the model intercept is close to the expected value."""
    expected_intercept = 0  # Since the line passes through the origin
    assert abs(model.intercept_ - expected_intercept) < 1e-5

def test_model_prediction():
    """Test if the model makes correct predictions on known data."""
    X_test = np.array([[6], [7]])
    y_expected = np.array([12, 14])  # Since y = 2x
    y_pred = model.predict(X_test)
    
    np.testing.assert_array_almost_equal(y_pred, y_expected, decimal=5)

