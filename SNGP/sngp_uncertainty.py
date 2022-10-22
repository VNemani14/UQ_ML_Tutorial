import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from dkl import DKL, GP, initial_values
from sngp import Laplace
from fc_resnet import FCResNet

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set(font_scale=1.6)
sns.set_theme(style='white')
sns.set_palette("colorblind")

np.random.seed(1)

def math_fun(x):
    x1=x[:,0]
    x2=x[:,1]
    g=((1.5+x1)**2+4)*(1.5+x2)/20-np.sin(5*(1.5+x1)/2)
    return g

### Prepare the training and test data for training the ML model
def make_data():
    mean_c1 = (2, 1.5)
    cov_c1 = [[0.3, 0], [0, 0.4]]
    c1_sample_no = 500
    x_c1 = np.random.multivariate_normal(mean_c1, cov_c1, (c1_sample_no, 1)).reshape(c1_sample_no, 2)

    mean_c2 = (-2.5, -2.5)
    cov_c2 = [[0.3, -0.2], [-0.2, 0.8]]
    c2_sample_no = 500
    x_c2 = np.random.multivariate_normal(mean_c2, cov_c2, (c2_sample_no, 1)).reshape(c2_sample_no, 2)

    mean_ood = (-7, -7.5)
    cov_ood = [[0.2, 0], [0, 0.2]]
    ood_sample_no = 200
    x_ood = np.random.multivariate_normal(mean_ood, cov_ood, (ood_sample_no, 1)).reshape(ood_sample_no, 2) 

    y_c1 = math_fun(x_c1).reshape(-1, 1)
    y_c2 = math_fun(x_c2).reshape(-1, 1)
    y_ood = math_fun(x_ood).reshape(-1, 1)

    ## Create train and test data for each cluster of data
    random_state = 1
    x_c1_train, x_c1_test, y_c1_train, y_c1_test = train_test_split(x_c1, y_c1, test_size=0.2, random_state=random_state)
    x_c2_train, x_c2_test, y_c2_train, y_c2_test = train_test_split(x_c2, y_c2, test_size=0.2, random_state=random_state)

    x_train = np.concatenate((x_c1_train, x_c2_train), axis = 0)
    x_test = np.concatenate((x_c1_test, x_c2_test), axis = 0)
    y_train = np.concatenate((y_c1_train, y_c2_train), axis = 0)
    y_test = np.concatenate((y_c1_test, y_c2_test), axis = 0)

    plt.figure(figsize=(10, 10))
    plt.scatter(x_c1_train[:, 0], x_c1_train[:, 1], color = 'red', marker = '+', label='1st cluster train')
    plt.scatter(x_c1_test[:, 0], x_c1_test[:, 1], color = 'red', marker = 's', label='1st cluster test')

    plt.scatter(x_c2_train[:, 0], x_c2_train[:, 1], color = 'blue', marker = '+', label='2nd cluster train')
    plt.scatter(x_c2_test[:, 0], x_c2_test[:, 1], color = 'blue', marker = 's', label='2nd cluster test')

    plt.scatter(x_ood[:, 0], x_ood[:, 1], color = 'purple', label = 'OOD samples')

    plt.xlabel('X1', fontsize=22, fontweight='bold')
    plt.ylabel('X2', fontsize=22, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    plt.legend(fontsize=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('train_test_data.pdf')

    return x_train, x_test, y_train, y_test, x_ood, y_ood

#### Create figures to visualize the uncertainty produced the SNGP model
def plot_uncertainty_map(x_mesh, output_std, process=False, contour=False):
    std_scaled = output_std/max(output_std)
    if not spectral_normalization:
        filename = 'DNN_GP_uncertainty'
    else:
        filename = 'SNGP_uncertainty'

    if process:
        std_scaled[np.where(std_scaled > 0.7)] = 0.9
        filename += '_processed'

    plt.figure(figsize=(10, 10))
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.scatter(x_mesh[:, 0], x_mesh[:, 1], rasterized = True, c = std_scaled, cmap = plt.get_cmap('viridis'))
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=28)
    plt.scatter(x_train[:, 0], x_train[:, 1], color = 'magenta', s = 20, alpha = 0.7, label = 'Training')
    plt.scatter(x_ood[:, 0], x_ood[:, 1], color = 'red', s = 20, alpha = 0.7, label = 'OOD')

    if contour:
        n = np.ceil(np.sqrt(len(std_scaled))).astype(int)
        h = std_scaled.reshape(n, n)
        contours = plt.contour(x_mesh[:, 0].reshape(n, n), x_mesh[:, 1].reshape(n, n), h)
        plt.clabel(contours, inline=True, fontsize=12)

        filename += '_with_contour'

    plt.legend(fontsize=20, loc=4)
    plt.xlabel(r'$x_1$', fontsize=40, fontweight='bold')
    plt.ylabel(r'$x_2$', fontsize=40, fontweight='bold')
    plt.xticks(fontsize=36, fontweight='bold')
    plt.yticks(fontsize=36, fontweight='bold')

    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect('equal')

    #ax.spines['top'].set_visible(False)
    plt.savefig(filename + '.pdf', bbox_inches='tight')

x_train, x_test, y_train, y_test, x_ood, y_ood = make_data()

torch.manual_seed(0)
n_samples = x_train.shape[0]
batch_size = 128

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

epochs = 5000
print(f"Training with {n_samples} datapoints for {epochs} epochs")

### SNGP model training and performance test
input_dim = 2
features = 128
depth = 6
num_outputs = 1 # regression with 1D output
spectral_normalization = False
coeff = 0.95
n_power_iterations = 1
dropout_rate = 0.01

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

#### Create meshes to verify the uncertainty of SNGP
n_meshes = 200
x1, x2 = np.meshgrid(np.linspace(-9, 5, n_meshes), np.linspace(-9, 5, n_meshes))
x_mesh = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
            
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
    xx = torch.tensor(x_mesh).float()
    if torch.cuda.is_available():
        xx = xx.cuda()
    pred = model(xx)

    output = pred[0].squeeze().cpu()
    output_var = pred[1].diagonal()
    output_std = output_var.sqrt().cpu()

plot_uncertainty_map(x_mesh, output_std, False, False)
plot_uncertainty_map(x_mesh, output_std, False, True)
plot_uncertainty_map(x_mesh, output_std, True, False)

