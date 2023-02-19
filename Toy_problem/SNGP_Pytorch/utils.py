import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.patches import Rectangle

sns.set(font_scale=1.6)
sns.set_theme(style='white')
sns.set_palette("colorblind")

np.random.seed(1)

def math_fun(x):
    x1=x[:,0]
    x2=x[:,1]
    g=((1.5+x1)**2+4)*(1.5+x2)/20-np.sin(5*(1.5+x1)/2)
    return g

### Prepare the training, testing and OOD data for the ML model
def make_data():
    mean_c1 = (8, 3.5)
    cov_c1 = [[0.4, -0.32], [-0.32, 0.4]]
    c1_sample_no = 500
    x_c1 = np.random.multivariate_normal(mean_c1, cov_c1, (c1_sample_no, 1)).reshape(c1_sample_no, 2)

    mean_c2 = (-2.5, -2.5)
    cov_c2 = [[0.4, -0.32], [-0.32, 0.4]]
    c2_sample_no = 500
    x_c2 = np.random.multivariate_normal(mean_c2, cov_c2, (c2_sample_no, 1)).reshape(c2_sample_no, 2)

    mean_ood = (-10, -7.5)
    cov_ood = [[0.2, -0.16], [-0.16, 0.2]]
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
    plt.scatter(x_c1_test[:, 0], x_c1_test[:, 1], color = 'red', marker = 'o', s = 8, label='1st cluster test')

    plt.scatter(x_c2_train[:, 0], x_c2_train[:, 1], color = 'blue', marker = '+', label='2nd cluster train')
    plt.scatter(x_c2_test[:, 0], x_c2_test[:, 1], color = 'blue', marker = 'o', s = 8, label='2nd cluster test')

    plt.scatter(x_ood[:, 0], x_ood[:, 1], color = 'purple', label = 'OOD samples')

    plt.xlabel('X1', fontsize=22, fontweight='bold')
    plt.ylabel('X2', fontsize=22, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    plt.legend(fontsize=20)
    plt.axis('equal')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.tight_layout()
    ax = plt.gca()
    ax.add_patch(Rectangle((-5, -5),
                        5, 5,
                        fc='none',
                        color ='green',
                        linewidth = 3))
    
    ax.add_patch(Rectangle((5.5, 1),
                    5, 5,
                    fc='none',
                    color ='green',
                    linewidth = 3))

    plt.savefig('train_test_data.pdf')

    n_meshes = 100
    x1, x2 = np.meshgrid(np.linspace(-5, 0, n_meshes), np.linspace(-5, 0, n_meshes))
    x_mesh_1 = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
    y_mesh_1 = math_fun(x_mesh_1).reshape(-1, 1).flatten()

    n_meshes = 100
    x1, x2 = np.meshgrid(np.linspace(5.5, 10.5, n_meshes), np.linspace(1, 6, n_meshes))
    x_mesh_2 = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
    y_mesh_2 = math_fun(x_mesh_2).reshape(-1, 1).flatten()

    x_mesh = np.concatenate((x_mesh_1, x_mesh_2), axis = 0)
    y_mesh = np.concatenate((y_mesh_1, y_mesh_2), axis = 0)

    n_meshes = 200
    x1, x2 = np.meshgrid(np.linspace(-15, 15, n_meshes), np.linspace(-15, 15, n_meshes))
    x_mesh_full = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis = 1)
    y_mesh_full = math_fun(x_mesh_full).reshape(-1, 1).flatten()

    return x_train, x_test, y_train, y_test, x_ood, y_ood, x_mesh, y_mesh, x_mesh_full, y_mesh_full

#### Create figures to visualize the uncertainty produced by the ML model
def plot_uncertainty_map(x_train, x_ood, x_mesh, output_std, filename, spectral_normalization = False, contour=False):
    std_scaled = output_std/max(output_std)

    plt.figure(figsize=(10, 10))
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.scatter(x_mesh[:, 0], x_mesh[:, 1], rasterized = True, c = std_scaled, cmap = plt.get_cmap('viridis'))
    
    if spectral_normalization:
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

    plt.savefig(filename + '.pdf', bbox_inches='tight')

def calculate_calibration(mean, std, y_ground_truth):
    if y_ground_truth.ndim > 1:
        y_ground_truth = y_ground_truth.flatten()

    expected_confidences = np.linspace(0, 1, num = 51)

    observed_confidences = []
    for expected_confidence in expected_confidences:
        intervals = scipy.stats.norm.interval(expected_confidence, loc=mean, scale=std)
        lb, ub = intervals[0], intervals[1]
        lb_ind = 1*(y_ground_truth >= lb)
        ub_ind = 1*(y_ground_truth <= ub)

        indicator = lb_ind * ub_ind
        observed_confidences.append(sum(indicator)/len(y_ground_truth))

    return expected_confidences, observed_confidences

def plot_calibration_curve(expected_confidences, observed_confidences, model_name):
    plt.figure(figsize=(10, 10))
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-', color='grey', 
             linewidth = 4, label='Ideal')
    plt.plot(expected_confidences, observed_confidences, '--', color='blue', 
            marker = 'o', markersize = 8, linewidth = 3, label = 'Calibration')

    plt.legend(fontsize=20, loc=4)
    plt.xlabel('Expected confidence', fontsize=40, fontweight='bold')
    plt.ylabel('Observed confidence', fontsize=40, fontweight='bold')
    plt.xticks(fontsize=36, fontweight='bold')
    plt.yticks(fontsize=36, fontweight='bold')

    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect('equal')

    plt.savefig(model_name + '_calibration_curve.pdf', bbox_inches='tight')




