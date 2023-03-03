## Toy problem:
Contains code for the toy example mentioned in Section 3.5 of the paper. 

The functional relationship between the output $y$ and the input vector $\textbf{x}=[x_1, x_2]$ is expressed as

$$ y(\textbf{x}) = \frac{1}{20}((1.5+x_1)^2+4)\times(1.5+x_2)-sin \frac{5\times(1.5+x_1)}{2}$$

- `Implement_GP_MC_NNE.ipynb`: consists the implementation of GPR, MC dropout and neural network ensemble
- `Implement_SNGP_DNNGP.ipynb`: consists the implementation of SNGP and DNN-GP. Toggle between SNNGP and DNNGP using the `spectral_normalization` variable.
- `SNGP_Pytorch`: SNGP implementation using PyTorch
- `Other_implementation`: Similar code in .py format

The uncertainty maps of the UQ methods on this regression toy problem look as follows

<p align="center">
  <img src="https://user-images.githubusercontent.com/94071944/219909512-1e2065b1-79d7-4eb9-b4e8-bd200c63415b.png" height="408" alt="capacity_curves" />
</p>
