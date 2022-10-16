# Uncertainty Quantification in Machine Learning for Engineering Design and Health Prognostics: A Comprehensive Review
(Code repository for the above titled review paper)

The goal of the study is to compare several ML models and their uncertainty quantification capability for Engineering Design and Prognostics

Methods explored are:
- Gaussian Process Regression (GP): (add citation)
- Deep Ensemble (DE): [Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).](https://doi.org/10.48550/arXiv.1612.01474)
- Bayesian Neural Networks (BN): (add citation)
- Monte Carlo Dropout (MC): [Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.](https://doi.org/10.48550/arXiv.1506.02142)
- Spectral-normalized Neural Gaussian Process (SNGP): (add citation)

These methods are evaluated on three different case studies. 

### Case Study 1: Design case study (Design)

(TBD)

### Case Study 2: Battery early life prediction (PHM)
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

![image](https://user-images.githubusercontent.com/94071944/196048824-a9ad0151-fcb8-4b66-97cb-88b125c6c538.png)

`Case_Study_2\Dataset` consists of
- `cycle_lives`: which has cycle lives of all of the cells
- `discharge_capacity`: which contains the capacity trajectory for all the cells. The file format is a `.csv` with columns `cycle number`, `capacity`, `initial capacity`
- `V_Q_curve`: with information of VQ curves starting from cycle 2 to 150. Each cell's data is stored as a `.csv` of size 1000x149 where the 1000 rows represent linearly spaced points from 3.5V to 2.0V (We use ***VQ(cycle=100)-VQ(cycle=10)*** to determine the total cycle life)

`Case_Study_2\UQ_models_train_evaluate.ipynb`: a simple to follow notebook that shows the implementation and evaluation of all the uncertainty quantification models

### Case Study 3: Turbofan engine prognostics (PHM)

(To Luca)
