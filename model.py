import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (hours studied vs marks)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70])

model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")