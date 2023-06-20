import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class L2A(nn.Module):
    def __init__(self, h1 = 512):
        super().__init__()

        # create MLP layers
        self.fc1 = nn.Linear(h1, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)
        self.fcg = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_action, command):
        # concatenate
        x = torch.cat((prev_action, command), dim=1)
        # pass through xyz MLP
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        # pass through gripper MLP
        g = self.fc1(x)
        g = self.relu(g)
        g = self.fc2(g)
        g = self.relu(g)
        g = self.fcg(g)
        g = self.sigmoid(g)
        return y, g
    
    def loss(self, y, g, label):
        # calculate huber loss for xyzv
        xyzv_loss = 10 * nn.functional.huber_loss(y, label[:, :4])
        # calculate binary cross entropy loss for gripper
        gripper_loss = nn.functional.binary_cross_entropy(g.squeeze(1), label[:, 4])
        # return total loss
        return xyzv_loss + gripper_loss
    
    def get_action(self, prev_action, command, obs):
        # get action from model
        xyzv, g = self.forward(prev_action, command)
        prev_action = torch.cat((xyzv, g), dim=1)
        # convert to numpy
        xyz = xyzv.detach().cpu().numpy()[0,:3] * 0.01
        # format for RLBench, 9D vector (coords, rot_grip_action, ignore_collisions_action)
        curr_xyz = obs['gripper_pose'][:3]
        curr_quat = obs['gripper_pose'][3:]
        x = xyz[1] + curr_xyz[0]
        y = -xyz[0] + curr_xyz[1]
        z = xyz[2] + curr_xyz[2]
        xyz = [x, y, z]
        gripper = g.detach().cpu().numpy()[0]
        action = np.concatenate((xyz, curr_quat, gripper, [1]))
        return action, prev_action

if __name__ == '__main__':
    # create model
    model = L2A()