import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adamax, Nadam, SGD, Adam
import h5py
import datetime


def parse_function(shape_X, final_shape_X, shape_Y, mask_indices, label_indices, noise_aug = None):
    """parse tf records files into tensors

    Takes a TFrecord file, shapes and volume mask and write a sparse tensor.
    The sparse tensor later sorted and converted to dense to comply with convolution layers
    Args:
        shape_X (tuple): shape of the vectorized input before reshaping
        final_shape_X (tuple): shape of the final input volume
        shape_Y (int): # of labels / neurons in the final layer
        mask_indices (bool): 4D array that mark the locations of non-zero values
    Returns:
        function: A _parse_function that except only a signle TFrecord file, could be used within a dataset object
    """
    def _parse_function(example_proto):
        # shape_Y == None for the test set where we have no labels
        if shape_Y == None:
            keys_to_features = {'X': tf.io.FixedLenFeature((shape_X), tf.float32)}

        else:
            keys_to_features = {'X': tf.io.FixedLenFeature((shape_X), tf.float32),
                                'Y': tf.io.FixedLenFeature((shape_Y), tf.float32)}
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        # if defined using [noise_aug] add normal noise with mean = 0 and SD = [noise_aug]
        if not noise_aug == None:
            X_data = parsed_features['X'] + tf.random.normal(shape_X, mean=0.0, stddev=noise_aug, dtype=tf.dtypes.float32)
        else:
            X_data =  parsed_features['X']
        # build a sparse tensor from the parsed file -> reorder -> convert to dense
        # (TF Conv layer except only dense tensors)
        X = tf.sparse.SparseTensor(mask_indices, X_data, dense_shape = final_shape_X)
        X = tf.sparse.reorder(X)
        X = tf.sparse.to_dense(X)
        Y = parsed_features['Y'][label_indices]
        return X, Y
    return  _parse_function


def OptimTypeFunc(x, OpLr, OpMom):
    # Function to determine to get desiered optimezer
    return {
        0: Nadam(lr=OpLr),
        1: Adamax(lr=OpLr),
        2: SGD(lr=OpLr, momentum=OpMom),
        3: Adam(lr=OpLr, clipnorm=1.0),
        4: Nadam(lr=OpLr)
    }[x]

class TrainingPlot(Callback):
    #https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
    def __init__(self, filename):
        self.filename = filename
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.mae = []
        self.mse = []
        self.val_losses = []
        self.val_mae = []
        self.val_mse = []
        self.logs = []

    def on_epoch_begin(self, batch, logs=None):
        print('Evaluating: epoch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.mae.append(logs.get('mean_absolute_error'))
        self.mse.append(logs.get('mse'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_mae.append(logs.get('val_mean_absolute_error'))
        self.val_mse.append(logs.get('val_mse'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, val loss against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "Training loss")
            plt.plot(N, self.val_losses, label = "Validation loss")
            plt.title("Training and validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename[:-4] + '_loss.png')
            plt.close()

            plt.style.use("seaborn")
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.mae, label = "Training MAE")
            plt.plot(N, self.val_mae, label = "Validation MAE")
            plt.title("Training and validation MAE [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("MAE")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename[:-4] + '_MAE.png')
            plt.close()

            plt.style.use("seaborn")
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.mse, label = "Training MSE")
            plt.plot(N, self.val_mse, label = "Validation MSE")
            plt.title("Training and validation MSE [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("MSE")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename[:-4] + '_mse.png')
            plt.close()

def read_hdf5(filepath):
    # read mat files originated from matlab
    with h5py.File(filepath, "r") as f:
        mat = f['SM_feature'][()]
    return mat

