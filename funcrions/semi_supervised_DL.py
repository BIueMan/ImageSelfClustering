import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from funcrions.semi_supervised_learning import *

from funcrions.semi_supervised_DL import loss_function

class ConvPart(nn.Module):
    def __init__(self):
        super(ConvPart, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv4(x)), 2)
        return x


class FCPart(nn.Module):
    def __init__(self, input_size):
        super(FCPart, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, image):
        super(CNN, self).__init__()
        self.conv = ConvPart()
        x = self.conv(torch.rand(1, *image.shape))
        self.fc = FCPart(x.shape[1]*x.shape[2]*x.shape[3])

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc(x)
        return x
    
    def loss_func(self, x_loc, P, sigmas, S, label):
        phi_label_func = self.forward
        loss = torch.abs(loss_function(x_loc, P, sigmas, phi_label_func, S, label))
        return loss

def loss_function(x_loc, P, sigmas, phi_label_func, S, label):
    m, n, i, j, k = P.shape
    m_x, n_x = x_loc
    norm_scale = 1

    def phi_label(m, n):
        if S[m, n] == label:
            phi = 1                                   # if this label
        elif S[m, n] != 0:
            phi = 0                                   # if other label
        else:
            phi = phi_label_func(P[m, n, :, :, :])    # no label
        return phi
    phi_x = phi_label(m_x, n_x)

    x = P[m_x, n_x, :, :, :]
    m_n_combinations = list(itertools.product(range(m), range(n)))
    sum = torch.tensor([0])
    for combination in m_n_combinations:
        m_y, n_y = combination
        y = P[m_y, n_y, :, :, :]
        phi_y = phi_label(m_y, n_y)
        w_xy = torch.from_numpy(weight(x, y, [sigmas[m_x, n_x], norm_scale]))
        w_yx = torch.from_numpy(weight(y, x, [sigmas[m_y, n_y], norm_scale]))
        sum += (w_xy + w_yx)*(phi_x+phi_y)
    
    return sum