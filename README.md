# Uncertainty Quantification in Machine Learning for Engineering Design and Health Prognostics: A Tutorial
(Code repository for the above titled paper)

The goal of the study is to compare several ML models and their uncertainty quantification capability for Engineering Design and Prognostics

Methods explored are:
- Gaussian Process Regression (GP): [Rasmussen, Carl Edward, and Christopher KI Williams. "Gaussian processes in machine learning." Lecture notes in computer science 3176 (2004): 63-71.](https://gaussianprocess.org/gpml/)
- Neural Network Ensemble (NNE): [Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).](https://doi.org/10.48550/arXiv.1612.01474)
- Monte Carlo Dropout (MC): [Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.](https://doi.org/10.48550/arXiv.1506.02142)
- Spectral-normalized Neural Gaussian Process (SNGP): [Liu, Jeremiah, et al. "Simple and principled uncertainty estimation with deterministic deep learning via distance awareness." Advances in Neural Information Processing Systems 33 (2020): 7498-7512.](https://doi.org/10.48550/arXiv.2006.10108)

These methods are evaluated on two different case studies. 

## Case Studies:
UQ methods are applied to two case studies
- Case study 1: Battery early life prediction
- Case study 2: Turbofan engine prognostics

All the models are built upon a ResNet-like architecture as shown below
<p align="center">
  <img src="https://user-images.githubusercontent.com/94071944/219261221-ac562a0f-f41f-44db-8405-b16780c60de4.png" height="408" alt="UQ model architectures" />
</p>


### Case Study 1: Battery early life prediction
The dataset for this case study was adopted from 
- [Severson, Kristen A., et al. "Data-driven prediction of battery cycle life before capacity degradation." Nature Energy 4.5 (2019): 383-391.](https://doi.org/10.1038/s41560-019-0356-8)
- [Attia, Peter M., et al. "Closed-loop optimization of fast-charging protocols for batteries with machine learning." Nature 578.7795 (2020): 397-402.](https://doi.org/10.1038/s41586-020-1994-5)

The dataset can be summarized as follows
| Dataset     | No. of cells |
| ----------- | ----------- |
| Train       | 41        |
| Test1       | 43        |
| Test2       | 40        |
| Test3       | 45        |

<p align="center">
  <img src="https://user-images.githubusercontent.com/94071944/219260555-eaa88b4c-f47b-4302-82eb-1bc9365104e8.png" height="408" alt="capacity_curves" />
</p>

See the directory for further details. 

### Case Study 2: Turbofan engine prognostics (PHM)
The dataset for this case study was adopted from 
- [Arias Chao, Manuel, et al. "Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics." Data 6.1 (2021): 5.](https://doi.org/10.3390/data6010005)

<p align="center">
  <img src="https://user-images.githubusercontent.com/94071944/236640537-632d7e5b-baff-4e1b-9122-9639f179cb4e.png" height="308" alt="Schematic representation of the CMAPSS model" />
</p>


## Toy problem:
Contains code for the toy example mentioned in Section 3.5 of the paper. 

The functional relationship between the output $y$ and the input vector $\textbf{x}=[x_1, x_2]$ is expressed as

$$ y(\textbf{x}) = \frac{1}{20}((1.5+x_1)^2+4)\times(1.5+x_2)-sin \frac{5\times(1.5+x_1)}{2}$$

The uncertainty maps of the UQ methods on this regression toy problem look as follows

<p align="center">
  <img src="https://user-images.githubusercontent.com/94071944/219909512-1e2065b1-79d7-4eb9-b4e8-bd200c63415b.png" height="408" alt="capacity_curves" />
</p>
