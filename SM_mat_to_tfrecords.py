import tensorflow as tf
import pandas as pd
import numpy as np
import os
from SM_utils import read_hdf5
import nibabel as nib

## set relevant paths
data_dir = '/media/galia-lab/Data1/users/gidonl/kaggle/TReNDS_brain/trends-assessment-prediction'
nas_data_dir = '/media/galia-nas/workingdir/gidon_l/kaggle/TReNDS_brain/data'
#data_dir = '/media/data2/gidon_l/kaggel_trends/data'
output_data_dir_train = data_dir + '/compres_tfrecords_train'
output_data_dir_val = data_dir + '/compres_tfrecords_val'
output_data_dir_test = data_dir + '/compres_tfrecords_test'
for dir in [output_data_dir_train, output_data_dir_val, output_data_dir_test]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

# define parameters
volume_path_template_train =  nas_data_dir + '/fMRI_train/*.mat'
volume_path_template_test =  nas_data_dir + '/fMRI_test/*.mat'
labels = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

# read labels + datasets splits
subjects_labels = pd.read_csv(data_dir + '/train_scores.csv', index_col = 'Id')
subjects_labels['Id'] = subjects_labels.index.values
df_set_splits = pd.read_csv('sets_splits.csv', index_col=0)
df_set_splits.sort_index(inplace=True)

## load the labels (functional connectivity networks locations) + labels
# get the train/val subject list
train_subjects = df_set_splits.loc[df_set_splits['set_code'] == 'train',:].index.values
val_subjects = df_set_splits.loc[df_set_splits['set_code'] == 'validation',:].index.values
test_subjects = df_set_splits.loc[df_set_splits['set_code'] == 'test',:].index.values

# create a csv with subjects ids and labels
train_labels_pd = subjects_labels.loc[train_subjects, :]
val_labels_pd = subjects_labels.loc[val_subjects, :]

# fill nans with mean values
for col in train_labels_pd.columns[:-1]:
    nan_locs = np.isnan(train_labels_pd.loc[:,col].values)
    print('Training data column ', col, 'has ', nan_locs.sum(), 'nans ({:.2f}%).'.format((100 * nan_locs.sum() / nan_locs.__len__())))
    train_labels_pd.loc[nan_locs, col] = np.nanmean(train_labels_pd.loc[:, col].values)
    print('Filling values with the mean')

for col in val_labels_pd.columns[:-1]:
    nan_locs = np.isnan(val_labels_pd.loc[:,col].values)
    print('Training data column ', col, 'has ', nan_locs.sum(), 'nans ({:.2f}%).'.format((100 * nan_locs.sum() / nan_locs.__len__())))
    val_labels_pd.loc[nan_locs, col] = np.nanmean(val_labels_pd.loc[:, col].values)
    print('Filling values with the mean')

# standardize the Y values
Y_mean, Y_std = train_labels_pd.values.mean(axis=0), train_labels_pd.values.std(axis=0)
for i, col in enumerate(train_labels_pd.columns[:-1]):
    train_labels_pd.loc[:, col] = (train_labels_pd.loc[:, col].values - Y_mean[i]) / Y_std[i]
for i, col in enumerate(val_labels_pd.columns[:-1]):
    val_labels_pd.loc[:, col] = (val_labels_pd.loc[:, col].values - Y_mean[i]) / Y_std[i]

def npy_to_tfrecords(X, Y, output_file):
    #https://stackoverflow.com/questions/45427637/numpy-to-tfrecords-is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrec/45428167#45428167
    # write records to a tfrecords file
    #options = tf.io.TFRecordOptions(compression_type=tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter(output_file) #, options=options

    # Loop through all the features you want to write
    # Feature contains a map of string to feature proto objects
    feature = {}
    feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
    if not np.all(Y) == None:
        feature['Y'] = tf.train.Feature(float_list=tf.train.FloatList(value=Y.flatten()))

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize the example to a string
    serialized = example.SerializeToString()


    # write the serialized objec to the disk
    writer.write(serialized)
    writer.close()

# load the mask file:
mask_nifti = nib.load(data_dir + '/fMRI_mask.nii')
channels = 53
mask_volume = mask_nifti.get_fdata()
sample_indices = np.array(np.where(mask_volume == 1.))
indices_per_sample = sample_indices.shape[1] * channels
mask_indices = np.empty((4, indices_per_sample), dtype=np.int8)
mask_indices[:] = np.nan
mask_indices[:3, :] = np.tile(sample_indices, reps=(1,channels))
mask_indices[3,:] = np.repeat(np.arange(channels),sample_indices.shape[1])



# read mat files and write tf records
print('Start converting training files to "tfrecord" format')
for i, col in enumerate(train_labels_pd['Id'].values):
    output_file = output_data_dir_train + '/train_' + str(col) + '.tfrecord'
    if not os.path.isfile(output_file):
        path = volume_path_template_train.replace('*', str(col))
        temp_vol = np.transpose(read_hdf5(path), axes = (3,2,1,0))[mask_volume == 1.]
        npy_to_tfrecords(temp_vol, train_labels_pd.loc[col, labels].values, output_file=output_file)
    print('.', end='')
print('.')

print('Start converting training files to "tfrecord" format')
for i, col in enumerate(val_labels_pd['Id'].values):
    output_file = output_data_dir_val + '/val_' + str(col) + '.tfrecord'
    if not os.path.isfile(output_file):
        path = volume_path_template_train.replace('*', str(col))
        temp_vol = np.transpose(read_hdf5(path), axes = (3,2,1,0))[mask_volume == 1.]
        npy_to_tfrecords(temp_vol, val_labels_pd.loc[col, labels].values, output_file=output_file)
    print('.', end='')
print('.')

print('Start converting training files to "tfrecord" format')
for i, col in enumerate(test_subjects):
    output_file = output_data_dir_val + '/test_' + str(col) + '.tfrecord'
    if not os.path.isfile(output_file):
        path = volume_path_template_test.replace('*', str(col))
        temp_vol = np.transpose(read_hdf5(path), axes = (3,2,1,0))[mask_volume == 1.]
        npy_to_tfrecords(temp_vol, None, output_file=output_file)
    print('.', end='')
print('.')

## plot the voxel's value distribution for a single volume
import seaborn as sns
from matplotlib import pyplot as plt

ax = sns.distplot(temp_vol[...,:].flatten(), bins= 300, norm_hist = False, kde = False)
plt.xlabel('Voxel values', fontsize=16)
plt.ylabel('# of voxels', fontsize=16)
plt.savefig('/media/galia-nas/workingdir/gidon_l/kaggle/TReNDS_brain/plots/volume_masked_dist.png')
plt.close('all')

ages = subjects_labels.loc[:,'age'].values
ax = sns.distplot(ages, bins= 100, norm_hist = False, kde = False)
plt.xlabel('Age values', fontsize=14)
plt.ylabel('# of subjects', fontsize=14)
plt.savefig('/media/galia-nas/workingdir/gidon_l/kaggle/TReNDS_brain/plots/age_dist.png')
plt.close('all')
