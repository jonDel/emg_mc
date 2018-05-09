
# coding: utf-8

# # Importing modules

# In[1]:
import math
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential
import keras.backend as K
import nina_helper as nh

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
    temp_array = np.zeros((rows, cols))
    col_min = data_array.min(axis=0)
    col_max = data_array.max(axis=0)
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            temp_array[i][j] = (data_array[i][j]-col_min[j])/(col_max[j]-col_min[j])
    return temp_array

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
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

def evaluate_subjs(model, db1_data, sub_num=None):
    rg_idx = range(len(db1_data))
    if not sub_num:
        idx_list = rg_idx
    else:
        idx_list = np.random.choice(rg_idx, sub_num)
    for subj in idx_list:
        preds_test = model.evaluate(db1_data[subj]['X_test'],
                                    db1_data[subj]['Y_test'])
        print("Test Loss for subject "+str(subj+1)+" = " + str(preds_test[0]))
        print("Test Accuracy for subject "+str(subj+1) + " = " + str(preds_test[1]))


# # Creating model

# In[4]:

def CNN_semg(input_shape=(16,10,1), classes=53):
    model = Sequential()
    "block 1"
    "32 filters,  a row of the length of number of electrodes,  ReLU"
    model.add(Conv2D(filters=32, kernel_size=(1,10), strides=(1,1),
                     padding='same', name='conv1',
                     input_shape=input_shape))
    "block 2"
    "32 filters 3*3,  ReLU,  average pool 3*3"
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
                     padding='same', name='conv2'))
    model.add(Activation('relu', name='relu2'))
    model.add(AveragePooling2D((3,3), strides=(2,2), name='pool1'))
    "block 3"
    "64 filters 5*5,  ReLu,  average pool 3*3"
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1),
                     padding='same', name='conv3'))
    model.add(Activation('relu', name='relu3'))
    model.add(AveragePooling2D((3,3), strides=(1,1), name='pool2'))
    "block 4"
    "64 filters 5*1,  ReLU"
    model.add(Conv2D(filters=64, kernel_size=(5,1), strides=(1,1),
                     padding='same', name='conv4'))
    model.add(Activation('relu', name='relu4'))
    "block 5"
    "filters 1*1,  softmax loss"
    model.add(Conv2D(filters=32, kernel_size=(1,1), strides=(1,1),
                     padding='same', name='conv5'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu',    name='fc1'))
    model.add(Dense(classes, activation='softmax', name='fc2'))   
    return model
