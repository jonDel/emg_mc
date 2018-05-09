
# coding: utf-8

# # Developing the project especified in the proposal
# 
# - Layer 1: input layer - Dimensions: D (number of sensors or channels) X 400 (number of samples)
# - for dataset 1: D=10; for dataset 2 and 3: D=12
# - Layer 2: Convolutional layer 64
# - Layer 3: Convolutional layer 64
# - Layer 4: Convolutional layer 64
# - Layer 5: Convolutional layer 64
# - Droupout
# - Layer 6: Dense layer LSTM 128
# - Dropout
# - Layer 7: Dense layer LSTM 64
# - Layer 8: Softmax layer 50 classes
# - RMSProp update rule
# - mini batch gradient descent - size=100, learning rate=0.001, decay factor=0.9
# - dropout: p=0.5
# - Test set: 1/4
# - Training set: 3/4

# # Importing modules

# In[1]:

import time
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Add, Dense, Activation,    ZeroPadding2D, BatchNormalization, Flatten, Lambda,    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,    LSTM, Dropout, Reshape, TimeDistributed, Convolution2D
from keras.models import Model, load_model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

np.random.seed(1)


# # Defining useful functions

# In[2]:

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def max_min_normalization(data_array):
    rows = data_array.shape[0]
    cols = data_array.shape[1]
    
    temp_array = np.zeros((rows,cols))
    col_min = data_array.min(axis=0)
    col_max = data_array.max(axis=0)

    for i in range(0,rows,1):
        for j in range(0,cols,1):
            temp_array[i][j] = (data_array[i][j]-col_min[j])/(col_max[j]-col_min[j])
    return temp_array


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# # Defining NINA functions

# In[3]:

def get_db1_data(db1_path, subj_list=[]):
    db_data = []
    window_len = 20
    window_inc = 10
    info_dict = nh.db1_info()
    reps = info_dict['rep_labels']
    nb_test_reps = 3 
    if not subj_list:
        idxs = range(1, info_dict['nb_subjects']+1)
    else:
        idxs = subj_list
    for subject in idxs:
        # Get EMG, repetition and movement data, don't cap maximum length of rest
        data_dict = nh.import_db1(db1_path, subject)

        # Decide window length (200ms window, 100ms increment)

        # Create a balanced test - training split based on repetition number
        train_reps, test_reps = nh.gen_split_balanced(reps, nb_test_reps)
        # Normalise EMG data based on training set 
        emg_data = nh.normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :]) 
        # Window data: x_all data is 4D tensor [observation, time_step, channel, 1] for use with Keras
        # y_all: movement label, length: number of windows
        # r_all: repetition label, length: number of windows
        x_all, y_all, r_all = nh.get_windows(reps, window_len, window_inc,
                                             emg_data, data_dict['move'],
                                             data_dict['rep'])
        train_idx = nh.get_idxs(r_all, train_reps[0, :]) 
        test_idx = nh.get_idxs(r_all, test_reps[0, :]) 
        Y_train = nh.to_categorical(y_all[train_idx])
        X_train = x_all[train_idx, :, :, :]
        Y_test = nh.to_categorical(y_all[test_idx])
        X_test = x_all[test_idx, :, :, :]
        sub_dict = {'X_train': X_train, 'Y_train': Y_train,
                    'X_test': X_test, 'Y_test': Y_test}
        db_data.append(sub_dict)
    return db_data

def append_db1_data(db_data):
    sub_dict = {}
    for sbj in db_data:
        if not sub_dict:
            sub_dict['X_train'] = sbj['X_train']
            sub_dict['Y_train'] = sbj['Y_train']
            sub_dict['X_test'] = sbj['X_test']
            sub_dict['Y_test'] = sbj['Y_test']
        else:
            sub_dict['X_train'] = np.vstack((sub_dict['X_train'], sbj['X_train']))
            sub_dict['Y_train'] = np.vstack((sub_dict['Y_train'], sbj['Y_train']))
            sub_dict['X_test'] = np.vstack((sub_dict['X_test'], sbj['X_test']))
            sub_dict['Y_test'] = np.vstack((sub_dict['Y_test'], sbj['Y_test']))
    return sub_dict

def evaluate_subjs(db1_data, sub_num=None):
    rg_idx = range(len(db1_data))
    if not sub_num:
        idx_list = rg_idx
    else:
        idx_list = np.random.choice(rg_idx, sub_num)
    for subj in idx_list:
        preds_test  = model.evaluate(db1_data[subj]['X_test'],
                                     db1_data[subj]['Y_test'])
        print("Test Loss for subject "+str(subj+1)+              " = " + str(preds_test[0]))
        print("Test Accuracy for subject "+str(subj+1)+              " = " + str(preds_test[1]))


# # Creating model

# In[4]:

def DeepConvLSTM(x_shape, class_number, filters, lstm_dims,
                                regularization_rate=0.01,
                                metrics=['accuracy']):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115
    Parameters
    ----------
    x_shape : tuple
        Shape of the input dataset: (num_samples, num_timesteps, num_channels)
    class_number : int
        Number of classes for classification task
    filters : list of ints
        number of filters for each convolutional layer
    lstm_dims : list of ints
        number of hidden nodes for each LSTM layer
    learning_rate : float
        learning rate
    regularization_rate : float
        regularization rate
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.
    Returns
    -------
    model : Keras model
        The compiled Keras model
    """
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    output_dim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels, 1)))
    # reshape a 2 dimensional array per file/person/object into a
    # 3 dimensional array
    #model.add(
    #    Reshape(target_shape=(dim_length, dim_channels, 1)))
    for filt in filters:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, kernel_size=(3, 1), padding='same',
                          kernel_regularizer=l2(regularization_rate),
                          kernel_initializer=weightinit))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(dim_length, filters[-1] * dim_channels)))

    for lstm_dim in lstm_dims:
        model.add(Dropout(0.5))  # dropout before the dense layer
        model.add(LSTM(units=lstm_dim, return_sequences=True,
                       activation='tanh'))

    # set up final dense layer such that every timestamp is given one
    # classification
    model.add(
        TimeDistributed(
            Dense(units=output_dim, kernel_regularizer=l2(regularization_rate))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))

    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)
    return model


# # Loading and preparing data

# In[5]:

import nina_helper as nh

subject_list = [1]
db1_path = "/home/b40153/github/emg_mc_venv/emg_mc/datasets/db1"
db1_data = get_db1_data(db1_path, subject_list)


# # Compile and plot the model

# In[6]:

#x_shape, class_number, filters, lstm_dims
model = DeepConvLSTM((20,10,1), 53, [64]*4, [128, 64])
# checkpoint
filepath = "weights-improvement-epoch:{epoch:02d}-acc:{acc:.2f}-val_acc:{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='acc',
                             save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir="logs/{}".format(time.strftime("%d/%m/%Y--%H:%M:%S")),
                          write_images=True)
callbacks_list = [checkpoint, tensorboard]
model.summary()


# # Fit, train & evaluate the model

# In[7]:

s1 = get_db1_data(db1_path,[1])[0]
model.load_weights("deepconvlstm_best-weights.hdf5")
preds_test = model.evaluate(s1['X_test'], s1['Y_test'])
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))


# In[9]:

db1_data = get_db1_data(db1_path)
data_db1 = append_db1_data(db1_data)
print (data_db1['X_train'].shape)
print (data_db1['Y_train'].shape)
print (data_db1['X_test'].shape)
print (data_db1['Y_test'].shape)
print ('db1 data size: ', len(db1_data))


# In[ ]:

epochs=40
model.load_weights("deepconvlstm_best-weights.hdf5")
model.fit(data_db1['X_train'], data_db1['Y_train'], epochs=epochs, batch_size=100, validation_split=0.2,
          callbacks=callbacks_list, verbose=1)
preds_train = model.evaluate(data_db1['X_train'], data_db1['Y_train'])
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))
# TODO: evaluate the test set using pre-trained weights (in this case, 
# the model already uses the best weights obtained from the training phase?)
preds_test  = model.evaluate(data_db1['X_test'], data_db1['Y_test'])
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))


# In[ ]:

evaluate_subjs(db1_data, 10)

