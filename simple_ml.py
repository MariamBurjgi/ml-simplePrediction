# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create some example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (reshape to 2D array)
y = np.array([2, 4, 5, 4, 5])  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
