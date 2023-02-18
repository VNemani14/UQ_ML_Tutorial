# Uncertainty Quantification in Machine Learning for Engineering Design and Health Prognostics: A Comprehensive Review
(Code repository for the above titled review paper)

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

`Case_Study_1\Dataset` consists of
- `cycle_lives`: which has cycle lives of all of the cells
- `discharge_capacity`: which contains the capacity trajectory for all the cells. The file format is a `.csv` with columns `cycle number`, `capacity`, `initial capacity`
- `V_Q_curve`: with information of VQ curves starting from cycle 2 to 150. Each cell's data is stored as a `.csv` of size 1000x149 where the 1000 rows represent linearly spaced points from 3.5V to 2.0V (We use ***VQ(cycle=100)-VQ(cycle=10)*** to determine the total cycle life)

`Case_Study_1\UQ_models_train_evaluate_FinalSNGP.ipynb`: a simple to follow notebook that shows the implementation and evaluation of SNGP

`Case_Study_1\UQ_models_train_evaluate_Final-Resnet.ipynb`: a simple to follow notebook that shows the implementation and evaluation of GP, NNE and MC uncertainty quantification models. Also includes postprocessing/comparison of all the models.



These UQ models are compared in terms of
- RUL prediction accuracy (RMSE)
- Negative log-likelihood (NLL)
- Calibration curves and expected calibration error (ECE)


### Case Study 2: Turbofan engine prognostics (PHM)
To enable fast data processing we make use of the code of [https://github.com/mohyunho/N-CMAPSS_DL][mohyunho] that we can quickly call via a submodule:
```
git submodule update --init
cd N-CMAPSS_DL/
```

Then we can download the data which we store into a new folder named `N-CMAPPS` inside the submodule:
```
mkdir N-CMAPSS
wget https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
unzip 17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
cd 17.\ Turbofan\ Engine\ Degradation\ Simulation\ Data\ Set\ 2/
unzip data_set.zip 
mv 17.\ Turbofan\ Engine\ Degradation\ Simulation\ Data\ Set\ 2/data_set/* N-CMAPSS/
```

Finally we can use the utility functions defined in the submodule to create a downsampled version of the train and test data:
```
python3 N-CMAPSS_DL/sample_creator_unit_auto.py -w 1 -s 1 --test 0 --sampling 500
python3 N-CMAPSS_DL/sample_creator_unit_auto.py -w 1 -s 1 --test 1 --sampling 500
```
