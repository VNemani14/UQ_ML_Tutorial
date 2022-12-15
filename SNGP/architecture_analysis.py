import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(10, 10))
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


expected_confidences = np.linspace(0, 1, num = 51)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-', color='grey', 
                    linewidth = 4, label='Ideal')

for depth in [6, 4, 2, 1]:
    for features in [128, 64, 16]:
        filename = 'SNGP_' + str(depth) + '_' + str(features) 
        observed_confidences = np.load(filename + '.npy')

        plt.plot(expected_confidences, observed_confidences,  
                linewidth = 3, alpha = 0.8, label = filename)

        plt.legend(fontsize=16, loc=4)
        plt.xlabel('Expected confidence', fontsize=40, fontweight='bold')
        plt.ylabel('Observed confidence', fontsize=40, fontweight='bold')
        plt.xticks(fontsize=36, fontweight='bold')
        plt.yticks(fontsize=36, fontweight='bold')

        plt.tight_layout()
        ax = plt.gca()
        ax.set_aspect('equal')

plt.savefig('calibration_curve_comp.pdf', bbox_inches='tight')