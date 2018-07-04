"""Script for loading and evaluating a model previously trained."""
import numpy as np
import sys
from deepconvlstm import session_run

np.random.seed(1)

usage = """
    python {script} <subject number> <database>
    Ex: python {script} 10 database_1
"""
if len(sys.argv) < 3:
    print (usage.format(script=sys.argv[0]))
subject_number = int(sys.argv[1])
database = sys.argv[2]
print ("Evaluating subject {} from database {}...".format(subject_number, database))
deepconv = session_run.DeepConvLstm(subject_number, database)
sub_data = deepconv.prepare_data()
input_shape = sub_data[0].shape
dcl_model, _ = deepconv.get_model(input_shape[1:])
res = deepconv.load_pretrained(dcl_model)
if not res[0]:
    print ("No pretrained session available. Exiting now...")
    exit()
preds_test = dcl_model.evaluate(sub_data[2], sub_data[3])
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))
