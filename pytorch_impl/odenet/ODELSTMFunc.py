from .mLSTM import mLSTM
import torch.nn as nn

class ODELSTMFunc(nn.Module):

    def __init__(self, input_size, hidden_size, embed_size, output_size):
        super(ODELSTMFunc, self).__init__()
        self.net = mLSTM(input_size, hidden_size, embed_size, output_size)

    def forward(self, t, y):
        #[inp, hidden , cell] = y
        #if inp.shape[0] == 2:
            #res = self.net(inp[0], inp.view(-1,1)[1])
        #else:
            #res= self.compute(inp, hidden, cell)
        return self.compute(y, self.hidden, self.cell)

    def compute(self, y, h, cell):
        res, h, cell = self.net(y, h, cell)
        self.initialize_state(h,cell)
        return res

    def initialize_state(self , h, c):
        self.hidden = h
        self.cell =c