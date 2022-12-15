import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import pickle

from utils import *

np.random.seed(1)

x_train, x_test, y_train, y_test, x_ood, y_ood, x_mesh, y_mesh, x_mesh_full, y_mesh_full = make_data()

scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
x_train_norm, y_train_norm = scalerX.fit_transform(x_train), scalerY.fit_transform(y_train)
x_test_norm, y_test_norm = scalerX.transform(x_test), scalerY.transform(y_test)
x_ood_norm, y_ood_norm = scalerX.transform(x_ood), scalerY.transform(y_ood)
x_mesh_norm = scalerX.transform(x_mesh)
y_mesh_norm = scalerY.transform(y_mesh.reshape(-1, 1))

model_GP = GaussianProcessRegressor(kernel=1 * RBF(length_scale=0.001, length_scale_bounds=(1e-3, 1e2)), 
                                    n_restarts_optimizer=10)
model_GP.fit(x_train_norm, y_train_norm)

# save
with open('GP.pkl','wb') as f:
    pickle.dump(model_GP, f)

print ('Training done!')

model_name = 'GP_'
range_diff = [-1, 1, 3, 5]
colors = ['blue', 'orange', 'green', 'fuchsia', 'gold']
markers = ['*', 'd', 'o', 's', '8']

plt.figure(figsize=(10, 10))
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-', color='grey', 
         linewidth = 4, label='Ideal')

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
    y_mesh = np.concatenate((y_mesh_1, y_mesh_2), axis = 0).reshape(-1, 1)

    x_mesh_norm = scalerX.transform(x_mesh)
    mean, std = model_GP.predict(x_mesh_norm, return_std = True)
    mean = scalerY.inverse_transform(mean.reshape(-1, 1)).flatten()
    std = std*(scalerY.data_max_ - scalerY.data_min_)

    expected_confidences, observed_confidences = calculate_calibration(mean, std, y_mesh.flatten())
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

