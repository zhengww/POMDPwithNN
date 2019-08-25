import torch
import torch.nn as nn
from torch.autograd import Variable

"""
######################################################
#
#  -- NN model class --
#  Architecture:
#  Input layer --> Hidden recurrent layer --> Linear readout -- > belief
#                                         --> Linear with softmax -- > action 
######################################################   
"""
class rnn_bel(nn.Module):
    def __init__(self, input_size, hidden_size_bel, output_size_bel, num_layers):
        super(rnn_bel, self).__init__()

        self.input_size = input_size
        self.hidden_size_bel = hidden_size_bel
        self.output_size_bel = output_size_bel
        self.num_layers = num_layers
        # self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size, hidden_size_bel, batch_first=True)
        self.linear_bel = nn.Linear(hidden_size_bel, output_size_bel, bias=True)

    def forward(self, x, h_0 = None):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        if h_0 is None:
            h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size_bel))


        out, _ = self.rnn(x, h_0)
        out_bel = self.linear_bel(out)
        return out_bel, out


"""
-- NN model for action --
Architecture:
Input layer --> Hidden recurrent layer --> Linear readout -- > belief
                                       [--> Nonlinear layer -- > Linear with softmax -- > action]    
"""


class net_act(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_act, softmax_temp = 1):
        super(net_act, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size_act = output_size_act

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size_act, bias=True)
        self.softmax = nn.Softmax(dim=2)

        self.softmax_temp = softmax_temp

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(x.size(0), self.hidden_size))

        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.softmax(out / self.softmax_temp)
        return out

