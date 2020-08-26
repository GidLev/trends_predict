seed = None#2020
import tensorflow as tf
if not seed == None:
    tf.random.set_seed(seed)
import numpy as np
if not seed == None:
    np.random.seed(seed)
from tensorflow.keras import backend as K
from SM_utils import parse_function
import pandas as pd
import random
import  nibabel as nib
import os
K.set_image_data_format('channels_last')

## set relevant paths
data_dir = '/media/galia-lab/Data1/users/gidonl/kaggle/TReNDS_brain/trends-assessment-prediction'

# define parameters
model_params = {'convFilt': [16, 32,64,128,64],
                'dropout': 0.35, 'l2': 0.0001,
                'optlr': 0.00001, 'optType': 3, 'optMom': 0.9} # 'optlr': 0.00015}
params = {'train_volume_path_template': '/compres_tfrecords_train/train_*.tfrecord',
          'val_volume_path_template': '/compres_tfrecords_val/val_*.tfrecord',
          'test_volume_path_template': '/compres_tfrecords_test/test_*.tfrecord',
          'model_path': '/media/galia-lab/Data1/users/gidonl/kaggle/models', # '/media/data2/gidon_l/kaggel_trends/models',#
          'batch_size': 16, 'epochs': 100, 'image_dim': (53,  63, 52, 53), 'labels_dim': (5,), #8
          'labels': ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'],
          'label_slices': slice(0,1)} #


# read labels + datasets splits
subjects_labels = pd.read_csv(data_dir + '/train_scores.csv', index_col = 'Id')
subjects_labels['Id'] = subjects_labels.index.values
df_set_splits = pd.read_csv('sets_splits.csv', index_col=0)
df_set_splits.sort_index(inplace=True)

## load the labels (functional connectivity networks locations) + labels
# get the train/val subject list
train_subjects = df_set_splits.loc[df_set_splits['set_code'] == 'train',:].index.values
val_subjects = df_set_splits.loc[df_set_splits['set_code'] == 'validation',:].index.values

# get the standardize parameters of the Y values
Y_mean = np.array([49.97658558, 51.46377466, 59.32577785, 47.33707329, 51.86560653])
Y_std = np.array([13.52275152, 9.84229405,10.9957966 ,11.03460485, 11.91871804,])
train_subjects = train_subjects
val_subjects = val_subjects

# load the mask file:
mask_nifti = nib.load(data_dir + '/fMRI_mask.nii')
channels = 53
mask_volume = mask_nifti.get_fdata()
sample_indices = np.array(np.where(mask_volume == 1.))
indices_per_sample = sample_indices.shape[1] * channels
mask_indices = np.empty((4, indices_per_sample), dtype=np.int8)
mask_indices[:] = np.nan
mask_indices[:3, :] = np.repeat(sample_indices, channels).reshape((3, sample_indices.shape[1] * channels))
mask_indices[3,:] = np.tile(np.arange(channels),sample_indices.shape[1])
mask_indices = mask_indices.T

# calculate y and x shapes
shape_X = (np.prod([sample_indices.shape[1], channels]),)
shape_X_final = params['image_dim']
shape_Y = params['labels_dim']
# get the input file list (tf records files)
file_names_train = [data_dir + params['train_volume_path_template'].replace('*', str(x)) for x in train_subjects[:2750]]
file_names_train = file_names_train + [data_dir + params['train_volume_path_template'].replace('*', str(x)) for x in train_subjects[2750:]]
file_names_val = [data_dir + params['val_volume_path_template'].replace('*', str(x)) for x in val_subjects[:1000]]
file_names_val = file_names_val + [data_dir + params['val_volume_path_template'].replace('*', str(x)) for x in val_subjects[1000:]]
if not seed == None:
    random.Random(seed).shuffle(file_names_train), random.Random(seed).shuffle(file_names_val)

## load a model:
model_path = '/media/galia-lab/Data1/users/gidonl/kaggle/models/ensemble_single_no_noise/baseline_2020-07-16_20-08-19/saved-model_66.best.hdf5'
if not os.path.isfile(model_path):
    raise Exception(model_path, 'was not found.')
else:
    model = tf.keras.models.load_model(model_path)

## evaluate the predictions

val_subjects_shuff = [int(x[86:91]) for x in file_names_val]
val_labels_pd = subjects_labels.loc[val_subjects_shuff, :]
#remove nans
for col in val_labels_pd.columns[:-1]:
    nan_locs = np.isnan(val_labels_pd.loc[:,col].values)
    print('Training data column ', col, 'has ', nan_locs.sum(), 'nans ({:.2f}%).'.format((100 * nan_locs.sum() / nan_locs.__len__())))
    val_labels_pd.loc[nan_locs, col] = np.nanmean(val_labels_pd.loc[:, col].values)
    print('Filling values with the mean')
# standardize the Y values
for i, col in enumerate(val_labels_pd.columns[:-1]):
    val_labels_pd.loc[:, col] = (val_labels_pd.loc[:, col].values - Y_mean[i]) / Y_std[i]


with tf.device('/cpu:0'):
    val_ds = tf.data.TFRecordDataset(file_names_val)
    # Parse the record into tensors.
    _parse_function_train = parse_function(shape_X, shape_X_final, shape_Y, mask_indices, params['label_slices'])
    val_ds = val_ds.map(_parse_function_train)
    # Repeat the input (One can consider first repeat than shuffle)
    steps_per_epoch = len(val_subjects) // params['epochs'] + 1
    val_ds = val_ds.repeat(1)
    # Generate batches
    val_ds = val_ds.batch(params['batch_size'])

print('Start evaluating...')
predicted_val = model.predict(val_ds)

from scipy import stats
target_labels = val_labels_pd.values[:,params['label_slices']]
for i, col in enumerate(val_labels_pd.columns[:-1].values[params['label_slices']]):
    print(col,':')
    print('Pearson correlation predicted-observed (rho, p):', stats.pearsonr(predicted_val[:,0],val_labels_pd.loc[:, col]))
    predicted_val_unnorm = (predicted_val * Y_std[i]) + Y_mean[i]
    target_labels_unnorm = (target_labels * Y_std[i]) + Y_mean[i]
    print('NEA: {:.3f}'.format(np.sum(np.abs(target_labels_unnorm - predicted_val_unnorm)) / target_labels_unnorm.sum()))
    print('MEA: {:.3f}'.format(np.abs(target_labels_unnorm - predicted_val_unnorm).mean()))
print('done')
