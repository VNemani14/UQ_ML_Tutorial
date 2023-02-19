### Case Study 1: Battery early life prediction

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
