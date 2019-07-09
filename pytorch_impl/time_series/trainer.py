import torch.optim as optim
from odenet.RunningAverageMeter import RunningAverageMeter
import time
import torch
import numpy as np
from Visualize import Visualize
#from torchdiffeq import odeint_adjoint as odeint

from torchdiffeq import odeint

def get_batch(true_y, batch_time, batch_size, t):
    data_size = true_y.shape[0]
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def train(model,y0, yt, t, niters, test_freq, dirName, batch_size, batch_time,op, saveImage=False):
    ii=0
    if op =='adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    #
    end = time.time()
    if saveImage:
        viz = Visualize(dirName)
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(yt,batch_time,batch_size, t)
        pred_y = odeint(model, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward(retain_graph=True)
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(model, y0, t)
                loss = torch.mean(torch.abs(pred_y - yt))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if saveImage:
                    viz.visualize(yt, pred_y, model, ii, t)
                ii += 1

        end = time.time()
    return pred_y, optimizer, loss, model