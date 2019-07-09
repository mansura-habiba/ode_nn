from .ODEGRU import ODEGRU
import torch.nn as nn

class ODEGRUFunc(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ODEGRUFunc, self).__init__()

        self.net = ODEGRU(input_size, hidden_size, output_size)

    def forward(self, t, y):
        h = self.net.hidden
        res, h = self.net(y,h)
       # res = res.flatten()
        return res
