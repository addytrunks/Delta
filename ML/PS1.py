import torch
import torch.nn as nn
import torch.optim as optim

# X is the independent variable, Y is the dependent variable
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([3, 6, 9, 12], dtype=torch.float32)

# Linear Regression model is setup to learn the relationship between X and Y.
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output.

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Define loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop where the neural network learns from the data, tries to improve its understanding, and gets a little better at its task
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward function : Makes predictions based on current parameter values
    outputs = model(X.view(-1, 1))
    loss = criterion(outputs, Y.view(-1, 1))  # Calculate the loss

    # Backward function and optimization: Updates the parameters of the neural network to minimize losses
    optimizer.zero_grad()
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the final weight
final_weight = model.linear.weight.item()
print(f'Final weight: {final_weight:.2f}')
