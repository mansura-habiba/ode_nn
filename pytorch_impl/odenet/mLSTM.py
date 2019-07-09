import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size):
        super(mLSTM, self).__init__()

        self.hidden_size = hidden_size
        # input embedding
        self.encoder = nn.Linear(input_size, embed_size)
        # lstm weights
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_om = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(embed_size, hidden_size)
        self.weight_ix = nn.Linear(embed_size, hidden_size)
        self.weight_cx = nn.Linear(embed_size, hidden_size)
        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # multiplicative weights
        self.weight_mh = nn.Linear(hidden_size, hidden_size)
        self.weight_mx = nn.Linear(embed_size, hidden_size)
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)


    def forward(self, inp, h_0, c_0):
        # encode the input characters
        inp = self.encoder(inp)
        # calculate the multiplicative matrix
        m_t = self.weight_mx(inp) * self.weight_mh(h_0)
        # forget gate
        f = self.weight_fx(inp) + self.weight_fm(m_t)
        f_g = torch.sigmoid(f)
        df = self.sigmoidPrime(f)
        # input gate
        i= self.weight_ix(inp) + self.weight_im(m_t)
        i_g = torch.sigmoid(i)

        di = self.sigmoidPrime(i)

        # output gate
        o= self.weight_ox(inp) + self.weight_om(m_t)
        o_g = torch.sigmoid(o)
        do =  self.sigmoidPrime(o)

        g_g = self.weight_cx(inp) + self.weight_cm(m_t)
        # intermediate cell state
        #c_tilda = torch.tanh(g_g)

        dc_tilda = self.tanhp(g_g)
        # current cell state
        #cx = f_g * c_0 + i_g * c_tilda
        cx = df * c_0 + di * dc_tilda
        # hidden state
        #hx = o_g * F.tanh(cx)
        hx = do * self.tanhp(cx)
        #out = self.decoder(hx.view(1,-1))
        out = self.decoder(hx)
        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size))
        c_0 = Variable(torch.zeros(1, self.hidden_size))
        return h_0, c_0

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def tanhp(self, input):
        return 1 - input ** 2

