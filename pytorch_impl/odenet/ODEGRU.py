import torch
import torch.nn as nn
from torch.autograd import Variable
import autograd.numpy as np
class ODEGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ODEGRU, self).__init__()
        self.input_units = input_size
        self.output_units = output_size
        self.hidden_units = hidden_size
        # input embedding
        #self.encoder = nn.Embedding(input_size, embed_size)
        # lstm weights
        self.weight_wz = nn.Linear(input_size, hidden_size)
        self.weight_uz = nn.Linear(hidden_size, hidden_size)

        self.weight_wr = nn.Linear(input_size, hidden_size)
        self.weight_ur = nn.Linear(hidden_size, hidden_size)

        self.weight_wy = nn.Linear(hidden_size, output_size)

        self.weight_wh = nn.Linear(input_size, hidden_size)
        self.weight_uh = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # h_0 = Variable(torch.zeros(1, self.hidden_units))
        h_0 = Variable(torch.zeros(1, self.hidden_units))
        return h_0

    def tanhp(self, input, deriv=False):
        if deriv:
            return 1 - input ** 2
        else:
            return torch.tanh(input)

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)



    def forward(self,  inp, h=None):
        if h is None:
            h= self.hidden
        # update gate
        z_g = torch.sigmoid(self.weight_wz(inp) + self.weight_uz(h))
        # reset gate
        r_g = torch.sigmoid(self.weight_wr(inp) + self.weight_ur(h))
        # hidden gate
        h1= (self.weight_wh(inp) + self.weight_uh(r_g * h))
        #dh = self.tanhp(h1, True)
        dh = torch.tanh(h1)

        # hidden state
        hx = dh *(1-z_g)#+ z_g#+ (dh * self.sigmoidPrime(z_g))
        o = self.decoder(hx)
        out =torch.sigmoid(o)
        #out = self.sigmoidPrime(out)
        #out = self.sigmoidPrime(o)
        self.hidden = hx
        return out, hx


