import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from matplotlib import pyplot as plt
from itertools import product
from Models_and_losses import *

from utils import *

## refer to the notebook here: https://github.com/Daniil-Selikhanovych/bnn-vi/blob/master/notebooks/Experiments.ipynb
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print (device)

x_train, x_test, y_train, y_test, x_ood, y_ood, x_mesh, y_mesh, x_mesh_full, y_mesh_full = make_data()

no_samples = x_train.shape[0]
n_epochs = 1000
n_layers = 4
BNN_VI = Bayesian_ReLU(2, 1, n_layers).to(device)
optbnn1 = optim.Adam(BNN_VI.parameters(), lr=1e-2)

train_model(BNN_VI, n_layers, optbnn1, elbo_loss, n_epochs, no_samples, 
            torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), variance = 'constant')

mu, std = make_predictions(BNN_VI, torch.Tensor(x_mesh_full).to(device), variance = 'constant')
mu, std = mu.cpu().detach().numpy(), std.cpu().detach().numpy()

plot_uncertainty_map(x_train, x_ood, x_mesh_full, std, 'VI_uncertainty', False, False)