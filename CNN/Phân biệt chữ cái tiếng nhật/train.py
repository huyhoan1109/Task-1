import torch
import torch as nn
import torch.optim as opt
from model import Net
from Dataloader import train_loader
import matplotlib.pyplot as plt

def train(net, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = clf_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 499:
                print(f'Epoch [{epoch+1}, {i}] Loss: {loss.item():.2f}')
    print('Done')

if __name__ == '__main__':
    clf_net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(clf_net.parameters(), lr=0.01, momentum=0.8)
    train(clf_net, criterion, optimizer)
    torch.save(clf_net.state_dict(),'net.pth')