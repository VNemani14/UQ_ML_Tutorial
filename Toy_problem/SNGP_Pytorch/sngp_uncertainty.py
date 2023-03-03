import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

from sklearn.metrics import mean_squared_error
import pandas as pd

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from dkl import DKL, GP, initial_values
from sngp import Laplace
from fc_resnet import FCResNet
from utils import *

np.random.seed(1)

x_train, x_test, y_train, y_test, x_ood, y_ood, x_mesh, y_mesh, x_mesh_full, y_mesh_full = make_data()

torch.manual_seed(0)
n_samples = x_train.shape[0]
batch_size = 128

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

epochs = 6000
print(f"Training with {n_samples} datapoints for {epochs} epochs")

### DNN-GP and SNGP model training and performance verification
input_dim = 2
features = 64
depth = 4
num_outputs = 1 # regression with 1D output
spectral_normalization = True
coeff = 0.95
n_power_iterations = 1
dropout_rate = 0.01

if spectral_normalization:
    model_name = 'SNGP_uncertainty'
else:
    model_name = 'DNNGP_uncertainty'

print (model_name)

feature_extractor = FCResNet(
    input_dim=input_dim, 
    features=features, 
    depth=depth, 
    spectral_normalization=spectral_normalization, 
    coeff=coeff, 
    n_power_iterations=n_power_iterations,
    dropout_rate=dropout_rate
)

num_gp_features = 1024
num_random_features = 1024
normalize_gp_features = True
feature_scale = 2
ridge_penalty = 1

model = Laplace(feature_extractor,
                features,
                num_gp_features,
                normalize_gp_features,
                num_random_features,
                num_outputs,
                len(ds_train),
                batch_size,
                ridge_penalty=ridge_penalty,
                feature_scale=feature_scale
                )

loss_fn = F.mse_loss
if torch.cuda.is_available():
    model = model.cuda()
    
lr = 1e-3
parameters = [
    {"params": model.parameters(), "lr": lr},
]   
optimizer = torch.optim.Adam(parameters)
pbar = ProgressBar()

def step(engine, batch):
    model.train()
    
    optimizer.zero_grad()
    
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def eval_step(engine, batch):
    model.eval()
    
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    y_pred = model(x)
            
    return y_pred, y

trainer = Engine(step)
evaluator = Engine(eval_step)

metric = Average()
metric.attach(trainer, "loss")
pbar.attach(trainer)

metric = Loss(lambda y_pred, y: F.mse_loss(y_pred[0], y))
metric.attach(evaluator, "loss")

@trainer.on(Events.EPOCH_COMPLETED(every=int(epochs/10) + 1))
def log_results(trainer):
    evaluator.run(dl_test)
    print(f"Results - Epoch: {trainer.state.epoch} - "
        f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
        f"Loss: {trainer.state.metrics['loss']:.2f}")
    
@trainer.on(Events.EPOCH_STARTED)
def reset_precision_matrix(trainer):
    model.reset_precision_matrix()

trainer.run(dl_train, max_epochs=epochs)
model.eval()

#### Create meshes to visualize the uncertainty of SNGP model
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
    xx = torch.tensor(x_mesh_full).float()
    if torch.cuda.is_available():
        xx = xx.cuda()
    pred = model(xx)

    mean = pred[0].squeeze().cpu()
    output_var = pred[1].diagonal()
    std = output_var.sqrt().cpu()

plot_uncertainty_map(x_train, x_ood, x_mesh_full, std, model_name, spectral_normalization)