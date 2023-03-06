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
