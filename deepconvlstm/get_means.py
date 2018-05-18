import re
import numpy as np

with open('deepconvlstm.log') as dfile:
    data = dfile.read()
all_tests = re.findall('Test Accuracy = (\d\.\d+)', data)
all_tests = [float(acc) for acc in all_tests]
mean = np.mean(all_tests)*100
std = np.std(all_tests)*100
print ('Results for {} subjects - Test accuracy: {:.4f}% +- {:.4f}%'.
       format(len(all_tests), mean, std))

