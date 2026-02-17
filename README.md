# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This experiment implements a feedforward neural network regression model using PyTorch.
The model accepts a single input feature and processes it through two hidden layers with ReLU activation functions to learn non-linear relationships.
The output layer predicts a continuous value.
The training process minimizes the Mean Squared Error (MSE) using the RMSProp optimizer, ensuring efficient convergence.

## Neural Network Model

<img width="1050" height="629" alt="image" src="https://github.com/user-attachments/assets/c7e8f77c-60b6-47ac-8996-55f66ec8feb4" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: vishwa v
### Register Number: 212224110062
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv('//content//Linear_Dataset (1).csv')

X = dataset[['INPUT']].values
y = dataset[['OUTPUT']].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


vishwa = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(vishwa.parameters(), lr=0.001)



def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")



train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")



loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_new = torch.tensor([[9]], dtype=torch.float32)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

prediction = ai_brain(X_new_tensor).item()

print(f"Prediction: {prediction}")
```
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
ai_brain = NeuralNet()
```
## Dataset Information

<img width="581" height="429" alt="image" src="https://github.com/user-attachments/assets/ea4473db-299a-459a-afc6-82eded44e387" />

## OUTPUT

<img width="530" height="238" alt="image" src="https://github.com/user-attachments/assets/a384220c-86c0-423c-994b-bcbd78b0efe2" />

### Training Loss Vs Iteration Plot

<img width="748" height="579" alt="image" src="https://github.com/user-attachments/assets/78f946b4-fe7f-4827-9806-b3420b492f13" />

### New Sample Data Prediction

<img width="372" height="35" alt="image" src="https://github.com/user-attachments/assets/89a1ce2e-d047-4351-8dba-f0795ae5eceb" />


## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
