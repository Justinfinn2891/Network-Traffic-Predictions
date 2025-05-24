import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and process dataset
dataset_path = './data/Midterm_53_group.csv'
dataset = pd.read_csv(dataset_path)

# Check the columns to ensure everything is correct
print(dataset.columns)

# Convert Time to a numerical value (hour + fraction of the hour)
dataset['Hour'] = pd.to_datetime(dataset['Time'], unit='s').dt.hour + pd.to_datetime(dataset['Time'], unit='s').dt.minute / 60

# Use Length as the traffic variable
X = dataset['Hour'].values.reshape(-1, 1).astype(np.float32)
y = dataset['Length'].values.reshape(-1, 1).astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# Define the model
class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # Linear regression model
    
    def forward(self, x):
        return self.linear(x)

model = RegressionNetwork()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    print("R² score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

# Plot
with torch.no_grad():
    full_pred = model(torch.from_numpy(X)).numpy()

plt.scatter(X, y, color='blue', label='Actual Traffic')
plt.plot(X, full_pred, color='red', label='Predicted Line')
plt.xlabel('Hour of Day')
plt.ylabel('Network Traffic (Length in Bytes)')
plt.legend()
plt.title('Network Traffic Prediction using Linear Regression (PyTorch)')
plt.show()