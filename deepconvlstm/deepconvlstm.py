# coding: utf-8
# # Developing the project especified in the proposal
# - Dataset 1 Layer 1: input layer - Dimensions: D (number of channels) X 20 (number of samples)
# - Datasets 2,3 Layer 1: input layer - Dimensions: D (number of channels) X 400 (number of samples)
# - for dataset 1: D=10; for dataset 2 and 3: D=12
# - Layer 2: Convolutional layer 64
# - Layer 3: Convolutional layer 64
# - Layer 4: Convolutional layer 64
# - Layer 5: Convolutional layer 64
# - Droupout
# - Layer 6: Dense layer LSTM 128
# - Dropout
# - Layer 7: Dense layer LSTM 64
# - Layer 8: Softmax layer 53 classes
# - RMSProp update rule
# - mini batch gradient descent - size=100, learning rate=0.001, decay factor=0.9
# - dropout: p=0.5
# - Test set: 1/4
# - Training set: 3/4
import numpy as np
from keras import optimizers
from keras.layers import Dense, Activation, BatchNormalization, Lambda, LSTM,\
    Dropout, Reshape, TimeDistributed, Convolution2D
from keras.models import Sequential
from keras.regularizers import l2
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
np.random.seed(1)


def model_deepconvlstm(x_shape, **kwargs):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    Parameters:
        x_shape (:obj:`tuple`):
            Shape of the input dataset: (num_samples, num_timesteps, num_channels)
        class_number (:obj:`int`,optional, *default* =53):
            Number of classes for classification task
        filters (:obj:`list`,optional, *default* =[64, 64, 64, 64]):
            number of filters for each convolutional layer
        lstm_dims (:obj:`list`,optional, *default* =[128, 64]):
            number of hidden nodes for each LSTM layer
        learn_rate (:obj:`float`,optional, *default* =0.001):
            learning rate
        reg_rate (:obj:`float`,optional, *default* =0.01):
            regularization rate
        metrics (:obj:`list`,optional, *default* =['accuracy']):
            List of metrics to calculate on the validation set.
            See https://keras.io/metrics/ for possible values.
        decay_factor (:obj:`float`,optional, *default* =0.9):
            learning rate decay factor
        dropout_prob (:obj:`float`,optional, *default* =0.5):
            dropout layers probability
        weight_init (:obj:`str`,optional, *default* ="lecun_uniform"):
            weights initialization function
        lstm_activation (:obj:`str`,optional, *default* ="tanh"):
            lstm layers activation function

    Returns
        model (:obj`object`):
            The compiled Keras model

    """
    def_args = {
        'class_number': 53,
        'filters': [64, 64, 64, 64],
        'lstm_dims': [128, 64],
        'learn_rate': 0.001,
        'decay_factor': 0.9,
        'reg_rate': 0.01,
        'metrics': ['accuracy'],
        'weight_init': 'lecun_uniform',
        'dropout_prob': 0.5,
        'lstm_activation': 'tanh'
    }
    def_args.update(kwargs)
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    output_dim = def_args['class_number']  # number of classes
    weight_init = def_args['weight_init']  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels, 1)))
    for filt in def_args['filters']:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, kernel_size=(3, 1), padding='same',
                          kernel_regularizer=l2(def_args['reg_rate']),
                          kernel_initializer=weight_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(dim_length, def_args['filters'][-1] * dim_channels)))
    for lstm_dim in def_args['lstm_dims']:
        model.add(Dropout(def_args['dropout_prob']))  # dropout before the dense layer
        model.add(LSTM(units=lstm_dim, return_sequences=True,
                       activation=def_args['lstm_activation']))
    # set up final dense layer such that every timestamp is given one
    # classification
    model.add(
        TimeDistributed(
            Dense(units=output_dim, kernel_regularizer=l2(def_args['reg_rate']))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))
    optimizer = optimizers.RMSprop(lr=def_args['learn_rate'],
                                   rho=def_args['decay_factor'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=def_args['metrics'])
    return model
