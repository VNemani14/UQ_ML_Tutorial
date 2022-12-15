from sngp import Laplace
from fc_resnet import FCResNet
import torch
import gpytorch
from utils import *

input_dim = 2
features = 64
depth = 4
num_outputs = 1 # regression with 1D output
spectral_normalization = True
coeff = 0.95
n_power_iterations = 1
dropout_rate = 0.01

if spectral_normalization:
    model_name = 'SNGP_'
else:
    model_name = 'DNN_GP_'

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

x_train, x_test, y_train, y_test, x_ood, y_ood, x_mesh, y_mesh, x_mesh_full, y_mesh_full = make_data()

torch.manual_seed(0)
n_samples = x_train.shape[0]
batch_size = 128

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

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

model.load_state_dict(torch.load(model_name[:-1]))
model.eval()

plt.figure(figsize=(10, 10))
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-', color='grey', 
            linewidth = 4, label='Ideal')

range_diff = [-1, 1, 3, 5]
colors = ['blue', 'orange', 'green', 'fuchsia', 'gold']
markers = ['*', 'd', 'o', 's', '8']
for diff in range_diff:
    n_meshes = 100
    x1, x2 = np.meshgrid(np.linspace(-5-diff, 0+diff, n_meshes), np.linspace(-5-diff, 0+diff, n_meshes))
    x_mesh_1 = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
    y_mesh_1 = math_fun(x_mesh_1).reshape(-1, 1).flatten()

    n_meshes = 100
    x1, x2 = np.meshgrid(np.linspace(5.5-diff, 10.5+diff, n_meshes), np.linspace(1-diff, 6+diff, n_meshes))
    x_mesh_2 = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
    y_mesh_2 = math_fun(x_mesh_2).reshape(-1, 1).flatten()

    x_mesh = np.concatenate((x_mesh_1, x_mesh_2), axis = 0)
    y_mesh = np.concatenate((y_mesh_1, y_mesh_2), axis = 0)

    #### Create meshes to visualize the uncertainty of SNGP model
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
        xx = torch.tensor(x_mesh).float()
        if torch.cuda.is_available():
            xx = xx.cuda()
        pred = model(xx)

        mean = pred[0].squeeze().cpu()
        output_var = pred[1].diagonal()
        std = output_var.sqrt().cpu()

    expected_confidences, observed_confidences = calculate_calibration(mean, std, y_mesh)
    index = range_diff.index(diff)
    plt.plot(expected_confidences, observed_confidences, '--', color = colors[index],
            marker = markers[index], markersize = 8, linewidth = 2, label = 'Case ' + str(index + 1))

plt.legend(fontsize=20, loc=4)
plt.xlabel('Expected confidence', fontsize=40, fontweight='bold')
plt.ylabel('Observed confidence', fontsize=40, fontweight='bold')
plt.xticks(fontsize=36, fontweight='bold')
plt.yticks(fontsize=36, fontweight='bold')

plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig(model_name + 'calibration_curves.pdf', bbox_inches='tight')

plt.figure(figsize=(10, 10))
ax = plt.gca()

i = 0
for diff in range_diff:
    print (colors[i])
    ax.add_patch(Rectangle((-5-diff/2, -5-diff/2),
                        5+diff, 5+diff,
                        fc='none',
                        color =colors[i],
                        linewidth = 3))
    
    ax.add_patch(Rectangle((5.5-diff/2, 1-diff/2),
                    5+diff, 5+diff,
                    fc='none',
                    color =colors[i],
                    linewidth = 3,
                    label = 'Case ' + str(i+1)))
    i = i + 1

plt.scatter(x_train[:, 0], x_train[:, 1], color = 'red', marker = '+', label='Training data')
plt.xlabel('X1', fontsize=40, fontweight='bold')
plt.ylabel('X2', fontsize=40, fontweight='bold')
plt.xticks(fontsize=36, fontweight='bold')
plt.yticks(fontsize=36, fontweight='bold')

plt.legend(fontsize=20)
plt.axis('equal')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.tight_layout()
plt.savefig('demo.pdf')

