import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

################################################################
num_layers = 3
hidden_neurons = 100
################################################################

batch_size = 16
learning_rate = 1e-3
epochs = 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

train_data = datasets.MNIST(
    "./",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    "./",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_neurons)
        
        self.act1 = nn.Sigmoid() ################################################################
        
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        
        self.act2 = nn.Sigmoid() ################################################################

        if num_layers==2:
            self.fc3 = nn.Linear(hidden_neurons, 10)
            self.act3 = nn.Identity()
            self.fc4 = nn.Identity()
        
        if num_layers == 3:
            self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
            
            self.act3 = nn.Sigmoid() ################################################################
            
            self.fc4 = nn.Linear(hidden_neurons, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        
        outputs = F.log_softmax(x, dim=1)
        return outputs

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = Net()     
optimizer = optim.Adam(params = model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()


for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n")
    train_loop(train_dataloader, model, loss, optimizer)
    test_loop(test_dataloader, model, loss)
print("Done!")