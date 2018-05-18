import matplotlib.pyplot as plt
import pickle
import sys

HIST_PATH = '../history/db1/'


def plot_result(subject):
    with open(HIST_PATH+'subject:{}_history.pickle'.format(subject), 'rb') as hfile:
        data = pickle.load(hfile)
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    subs = sys.argv[1]
    for subj in subs.replace(' ', '').split(','):
        plot_result(subj)
