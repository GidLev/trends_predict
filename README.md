# TReNDS Neuroimaging Kaggle challenge
Code for Kaggle's Trends neuroimaging challenge - predicting age with 4D group ICA volumes 

As part of the TReNDS challenge, we try to predict the age from spatial fMRI-based group ICA maps, as input, where every input consists of a 4D tensor which includes 53 channels of 3D spatial maps of size [52x63x53] voxels. We used an ensemble of multiple 3D CNNs (m = 10) each trained separately. As in previous work using deep CNNs (Levakov, Rosenthal, Shelef, Raviv, & Avidan, 2020) our models comprising the ensemble differ only in their random weight initialization, meaning they had identical architecture and were trained on the same samples. After each network was independently trained, a linear regression model for age prediction is learned from the outputs of the ten networks using 10-fold cross validation. 

## file descriptions
SM_mat_to_tfrecords.py - script for converting Numpy arrays to TF records

SM_training.py - training the model

SM_evaluate_prediction.py - calculate correlation, mean absolute error (MEA) and normalized  absolute error (NAE)

SM_utils.py - various  utility functions 

## Reference
Levakov, G., Rosenthal, G., Shelef, I., Raviv, T. R., & Avidan, G. (2020). From a deep learning model back to the brain—Identifying regional predictors and their relation to aging. Human Brain Mapping, 41(12), 3235–3252. https://doi.org/10.1002/hbm.25011
