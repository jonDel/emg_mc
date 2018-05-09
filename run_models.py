
# coding: utf-8
from os.path import isfile
import numpy as np
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import cnn_shallow as cnn

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
np.random.seed(1)
DB1_PATH = "/home/b40153/github/emg_mc_venv/emg_mc/datasets/db1"
#BEST_WEIGHTS = "best-weights.hdf5"
BEST_WEIGHTS = "best-weights_not_over.hdf5"


def get_cnn_model(file_weights):
    model = cnn.CNN_semg(input_shape=(20, 10, 1), classes=53)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(file_weights, verbose=1, monitor='val_acc',
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.summary()
    return (model, callbacks_list)

def train_cnn(epochs, dataset, file_weights="best-weights.hdf5"):
    model, callbacks_list = get_cnn_model(file_weights)
    if isfile(file_weights):
        print ('loading pre-trained file {}'.format(file_weights))
        model.load_weights(file_weights)
    model.fit(dataset['X_train'], dataset['Y_train'], epochs=epochs,
              batch_size=100, validation_split=0.2,
              callbacks=callbacks_list, verbose=1)
    return model


def evaluate_model(dataset, model, file_weights=None):
    if isfile(file_weights):
        model.load_weights(file_weights)
    preds_train = model.evaluate(dataset['X_train'], dataset['Y_train'])
    print("Train Loss = " + str(preds_train[0]))
    print("Train Accuracy = " + str(preds_train[1]))
    preds_test = model.evaluate(dataset['X_test'], dataset['Y_test'])
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))


if __name__ == "__main__":
    epochs = 5
    db1_data = cnn.get_db1_data(DB1_PATH)
    dataset = cnn.append_db1_data(db1_data)
    print (dataset['X_train'].shape)
    print (dataset['Y_train'].shape)
    print (dataset['X_test'].shape)
    print (dataset['Y_test'].shape)
    print ('db1 data size: ', len(db1_data))
    model = train_cnn(epochs, dataset)
    #model, _ = get_cnn_model(BEST_WEIGHTS)
    evaluate_model(dataset, model, BEST_WEIGHTS)
    cnn.evaluate_subjs(model, db1_data, 10)

