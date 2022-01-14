# Load package
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor

import math
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False

# Build LSTM cell
class LSTMcell(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, bias=True):
    
        super(LSTMcell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=bias)

        self.init_weight()

    def init_weight(self):

        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
  
    def forward(self, x, hidden):
      
        ht, ct = hidden
        x = x.view(-1,x.size(1))
        gates = self.W(x) + self.U(ht)

        forget_gate, input_gate, cell_gate, out_gate = gates.chunk(4, 1)

        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        cell_gate = forget_gate * ct + input_gate * torch.tanh(cell_gate) 
        out_gate =  torch.sigmoid(out_gate)

        h_af = out_gate * torch.tanh(cell_gate)
        return h_af, cell_gate

class LSTMnet(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, layer_dim=None, output_dim=None, bias=True):

        super(LSTMnet, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMcell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if cuda:
            h0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
            c0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            c0 = Tensor(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn, cn))
            outs.append(hn)
        
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out