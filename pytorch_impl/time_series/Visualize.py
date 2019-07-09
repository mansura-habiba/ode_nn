import os
import matplotlib.pyplot as plt
import numpy as np
import torch

class Visualize():
    def __init__(self, dirName, saveImage=False):
        super(Visualize, self).__init__()
        self.saveImage = saveImage
        if saveImage:
            self.makedirs(dirName)
        self.fig = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_traj = self.fig.add_subplot(131, frameon=False)
        self.ax_phase = self.fig.add_subplot(132, frameon=False)
        self.ax_vecfield = self.fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    def makedirs(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def visualize(self, true_y, pred_y, odefunc, itr, t):
        self.ax_traj.cla()
        self.ax_traj.set_title('Trajectories')
        self.ax_traj.set_xlabel('t')
        self.ax_traj.set_ylabel('x,y')
        self.ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        self.ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        self.ax_traj.set_xlim(t.min(), t.max())
        self.ax_traj.set_ylim(-2, 2)
        self.ax_traj.legend()

        self.ax_phase.cla()
        self.ax_phase.set_title('Phase Portrait')
        self.ax_phase.set_xlabel('x')
        self.ax_phase.set_ylabel('y')
        self.ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        self.ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        self.ax_phase.set_xlim(-2, 2)
        self.ax_phase.set_ylim(-2, 2)

        self.ax_vecfield.cla()
        self.ax_vecfield.set_title('Learned Vector Field')
        self.ax_vecfield.set_xlabel('x')
        self.ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        self.ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        self.ax_vecfield.set_xlim(-2, 2)
        self.ax_vecfield.set_ylim(-2, 2)

        self.fig.tight_layout()
        if self.saveImage:
            plt.savefig('png2/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)