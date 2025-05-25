import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

network_data_pathName = './data/Midterm_53_group.csv'
network_src_data = pd.read_csv(network_data_pathName)
network_src_data['Time_src'] = network_src_data['Time'] / 60

X_src = network_src_data['Time_src'].values.reshape(-1,1).astype(np.float32) #Turning them into 2d arrays
y_src = network_src_data['Length'].values.reshape(-1,1).astype(np.float32)


scaler_y = StandardScaler()
y_src_scaled = scaler_y.fit_transform(y_src)

X_train, X_test, y_train, y_test = train_test_split(X_src, y_src_scaled, test_size = 0.2, random_state= 42)

#Tensors used for training
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

network_model = RegressionNetwork()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network_model.parameters(), lr=0.001)

epochs = 1000
for i in range(epochs):
    network_model.train()
    optimizer.zero_grad()
    output_src = network_model(X_train_tensor)
    loss = criterion(output_src, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f'EPOCH: [{i + 1} / {epochs}], Loss: {loss.item():.4f}')

network_model.eval() #Stop training 

with torch.no_grad():
    full_pred_scaled = network_model(torch.from_numpy(X_src).float()).numpy().flatten()
    full_pred = scaler_y.inverse_transform(full_pred_scaled.reshape(-1, 1)).flatten()

sorted_indices = X_src[:, 0].argsort()
X_sorted = X_src[sorted_indices]
full_pred_sorted = full_pred[sorted_indices]

def predict_bytes(model, time_in_seconds, scaler):
    # Convert time to minutes, float32, and make it 2D tensor shape (1,1)
    time_in_minutes = np.array([[time_in_seconds / 60]], dtype=np.float32)
    time_tensor = torch.from_numpy(time_in_minutes)
    
    model.eval()
    with torch.no_grad():
        pred_scaled = model(time_tensor).numpy().reshape(-1, 1)
    
    # Inverse transform to get original byte scale
    pred_original = scaler.inverse_transform(pred_scaled)
    
    return pred_original.item()

input_time_seconds = 5000  # For example, 1 hour = 3600 seconds
predicted_bytes = predict_bytes(network_model, input_time_seconds, scaler_y)
print(f"Predicted network traffic bytes at {input_time_seconds} seconds (minute {input_time_seconds/60:.2f}): {predicted_bytes:.2f}")

plt.scatter(X_src, scaler_y.inverse_transform(y_src).flatten(), color='blue', label='Actual Traffic')
plt.plot(X_sorted, full_pred_sorted, color='red', label='Predicted Line')
plt.xlabel('Minute of Day')
plt.ylabel('Network Traffic (Length in Bytes)')
plt.legend()
plt.title('Network Traffic Prediction using Linear Regression (PyTorch)')
plt.show()

for name, param in network_model.named_parameters():
    print(f"{name}: {param.data.item():.6f}")