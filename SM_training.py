seed = None#2020
import tensorflow as tf
if not seed == None:
    tf.random.set_seed(seed)
import numpy as np
if not seed == None:
    np.random.seed(seed)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Flatten, Dense, SpatialDropout3D, Dropout, Add, Concatenate
from tensorflow.keras.layers import AveragePooling3D, MaxPooling3D, Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K
from SM_utils import OptimTypeFunc, parse_function, TrainingPlot, save_curr_script
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import pandas as pd
from tensorflow.keras.initializers import glorot_uniform
import random
import pickle
import  nibabel as nib
K.set_image_data_format('channels_last')

def STN(model_params, input_shape=(53,  52, 63, 53), output_shape = 1):
    '''Build the CNN model
    model_params: dict with the architecture parameters
        convFilt: list with number of filters in each layer
        dropout: dropout before the last dense layer
        l2: weights decay regularization parameter (L2)
        optlr: learning rate
        optType: 0 =  Nadam, 1 = Adamax, 2 = SGD, 3 = Adam, 4 = Nadam
        optMom: momentum rate
    input_shape: tuple, the shape of the input volume
    output_shape: integer, # of output neurons
    '''
    def convolutional_block(X, filters, stage, s=2, weight_decay = 0.001):
        # defining name basis
        conv_name_base = 'conv_' + stage
        bn_name_base = 'bn_' + stage
        # First component of main path
        X = Conv3D(filters, (3, 3, 3), strides=(1, 1, 1), name=conv_name_base + '_1', kernel_initializer=glorot_uniform(seed=0),
                   kernel_regularizer = regularizers.l2(weight_decay))(X)
        X = BatchNormalization(name=bn_name_base + '_1')(X)
        X = Activation('relu')(X)
        X = Conv3D(filters, (3, 3, 3), strides=(s, s, s), name=conv_name_base + '_2', kernel_initializer=glorot_uniform(seed=0),
                   kernel_regularizer = regularizers.l2(weight_decay))(X)
        X = BatchNormalization(name=bn_name_base + '_2')(X)
        X = Activation('relu')(X)
        return X

    filters = model_params.get('convFilt')
    weight_decay = model_params.get('l2')

    optType, optlr, optMom = model_params.get('optType'), model_params.get('optlr'), model_params.get('optMom')
    Optm = OptimTypeFunc(optType, optlr, optMom)
    X_input = Input(shape=input_shape)
    X = Conv3D(filters[0], (1, 1, 1), name='conv3D_first_reduce_channels', kernel_initializer=glorot_uniform(seed=0),
                    kernel_regularizer = regularizers.l2(weight_decay))(X_input)
    X = BatchNormalization(name='bn_f_reduce_channels')(X)
    X = Activation('relu')(X)
    X = convolutional_block(X, filters = filters[1], stage = 'a', weight_decay = weight_decay)
    X = convolutional_block(X, filters = filters[2], stage = 'b', weight_decay = weight_decay)
    X = Conv3D(filters[4], (1, 1, 1), name='conv3D_second_reduce_channels', kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=regularizers.l2(weight_decay))(X)
    X = BatchNormalization(name='bn_s_reduce_channels')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dropout(model_params.get('dropout'))(X)
    final_pred = Dense(output_shape)(X)
    model = Model(inputs=X_input, outputs=final_pred, name='SFCN')
    model.compile(loss=['mse'], optimizer=Optm, metrics=['mean_absolute_error', 'mse'])
    model.summary()
    return model

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

## plot labels cross-correlations:
import seaborn as sns
from matplotlib import pyplot as plt

def corrdot(*args, **kwargs):
    # create a cross-correlation plot
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

# sort the values in one dataframe for plotting
plot_df = subjects_labels.copy()
plot_df.set_index('Id', inplace=True)
plot_df.rename(columns={'age': 'Age', 'domain1_var1': 'Domain 1 variable 1', 'domain1_var2': 'Domain 1 variable 2',
                        'domain2_var1': 'Domain 2 variable 1', 'domain2_var2': 'Domain 2 variable 2'}, inplace=True)
# plot with seaborn
g = sns.PairGrid(plot_df, aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)
g.fig.subplots_adjust(bottom=0.1, left=0.1)
plt.savefig('/media/galia-nas/workingdir/gidon_l/kaggle/TReNDS_brain/plots/cross_corr_age_and_clinical_measures.png')  # , dpi=300
plt.show()

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


## define the dataset iterator (https://cs230.stanford.edu/blog/datapipeline/#best-practices):
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
with tf.device('/cpu:0'):
    train_ds = tf.data.TFRecordDataset(file_names_train, num_parallel_reads = AUTOTUNE) #
    # Shuffle the dataset
    train_ds = train_ds.shuffle(buffer_size=8, seed= seed)
    # Parse the record into tensors
    _parse_function_train = parse_function(shape_X, shape_X_final, shape_Y, mask_indices, params['label_slices'], noise_aug = None)
    train_ds = train_ds.map(_parse_function_train) #, num_parallel_reads = AUTOTUNE
    # Repeat the input (One can consider first repeat than shuffle)
    steps_per_epoch = (len(train_subjects) // params['batch_size']) + 1
    train_ds = train_ds.repeat(params['epochs'] * steps_per_epoch)
    # Generate batches
    train_ds = train_ds.batch(params['batch_size'])
    train_ds = train_ds.prefetch(2)

with tf.device('/cpu:0'):
    val_ds = tf.data.TFRecordDataset(file_names_val) #, num_parallel_reads = AUTOTUNE
    _parse_function_test = parse_function(shape_X, shape_X_final, shape_Y, mask_indices, params['label_slices'], noise_aug = None)
    val_ds = val_ds.map(_parse_function_test) #, num_parallel_calls=AUTOTUNE
    # Repeat the input (One can consider first repeat than shuffle)
    steps_per_epoch = (len(val_subjects) // params['batch_size']) + 1
    val_ds = val_ds.repeat(params['epochs']* steps_per_epoch)
    # Generate batches
    val_ds = val_ds.batch(params['batch_size'])


num_ensemble = 10
for i in np.arange(num_ensemble):
    print('start ensemble #', i)
    ## define path to save the model
    if params['model_path'].find('/baseline_' ) == -1:
        params['model_path'] = params['model_path'] + '/baseline_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    else:
        params['model_path'] = params['model_path'][:params['model_path'].find('/baseline_' )] + '/baseline_' + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    # define TF callbacks
    # saves the model after each evaluation improvement
    checkpoint = ModelCheckpoint(params['model_path'] + '/saved-model_{epoch:02d}.best.hdf5',
                                 verbose=1, save_weights_only=False, save_best_only=False, mode='min')
    callbacks = TrainingPlot(params['model_path'] + '/plot.png') # update the training plot after each training round
    tensorboard_callback = TensorBoard(log_dir=params['model_path']) # write to TensorBoard
    csv_logger = CSVLogger(params['model_path'] + '/model_history_log.csv', append=False) # save the model history
    callbacks_list = [checkpoint, callbacks,
                      tensorboard_callback, csv_logger]
    # build the model or load one if exist
    model = STN(model_params, input_shape = params['image_dim'], output_shape = len(params['labels'][params['label_slices']]))
    # start training
    model_history = model.fit(train_ds, epochs=params['epochs'], steps_per_epoch=(len(train_subjects) // params['batch_size']) + 1,
                                  callbacks=callbacks_list,
                                  validation_data=val_ds, validation_steps=(len(val_subjects) // params['batch_size']) + 1)
    # save history
    with open(params['model_path'] + '/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(model_history.history, file_pi)

print('Done training')
