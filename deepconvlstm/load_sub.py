# coding: utf-8
import sys
from prepare_data import get_deepconvlstm, prepare_data, DATASETS_DICT, load_pretrained

if __name__ == "__main__":
    timestep_num = 20
    inc_len = 100
    train_split = 3
    dataset = "dataset_1"
    classes = DATASETS_DICT[dataset]["dataset_info"]["nb_moves"]
    LEARN_RATE = 0.001
    w_folder = DATASETS_DICT[dataset]["weights_path"]
    subs = sys.argv[1]
    for subject_number in subs.replace(' ', '').split(','):
        print ('Evaluating subject {}...'.format(subject_number))
        sub_data = prepare_data(subject_number, timestep_num,
                                inc_len, train_split, dataset)
        input_shape = sub_data[0].shape
        model, _ = get_deepconvlstm(input_shape[1:], subject_number, classes, dataset)
        res = load_pretrained(model, w_folder, 'val_acc', subject_number)
        if res[0]:
            print ('Using pre-trained weights... evaluating from epoch {}'.format(res[1]))
            model = res[0]
        preds_test = model.evaluate(sub_data[2], sub_data[3])
        print("Test Loss = " + str(preds_test[0]))
        print("Test Accuracy = " + str(preds_test[1]))

