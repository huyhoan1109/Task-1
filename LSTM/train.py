import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as opt

from model import LSTMnet, cuda
from Dataloader import train_loader, test_loader, train_data

import matplotlib.pyplot as plt
import sys


def batch_scale(data):
    # Scale lại data cho 
    inputs, labels = data
    seq_dim = inputs.size()[-2]
    input_dim = inputs.size()[-1]
    if cuda:
        inputs = inputs.view(-1, seq_dim, input_dim).cuda() 
        labels = labels.cuda()
    else:
        inputs = inputs.view(-1, seq_dim, input_dim)

    return inputs, labels

def train(model, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
    
            inputs, labels = batch_scale(data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if i % 500 == 0:
            print(f'Epoch [{epoch+1}, {i}] Loss: {loss.item():.2f}')
    return model

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def make_map():

    label_maps = dict()
    class_num = len(train_data.classes)
    for i in range(class_num):
        label_maps[i] = train_data.classes[i]
    
    return label_maps, class_num

if __name__ == '__main__':

    label_maps, class_num = make_map()
    
    input_dim = 28
    hidden_dim = 128
    layer_dim = 1
    output_dim = class_num

    model = LSTMnet(input_dim, hidden_dim, layer_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(model.parameters(), lr=0.01, momentum=0.8)

    model = train(model, optimizer, criterion)
    save_model(model, 'Net.pth')