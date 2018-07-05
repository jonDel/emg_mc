# Deep learning for classifing sEMG signals

## Installation

### Cloning repository
```bash
git clone https://github.com/jonDel/emg_mc.git
```
### Installing environment
```bash
cd emg_mc
bash install_environ.sh
```
### Activating environment
```bash
source emg_mc_venv/bin/activate
```
### Downloading the datasets (takes a while, 22 Gb)
```bash
python scripts/datasets_download.py
```
## Checking results from pre-trained sessions

### Evaluating model with a test set from a subject
```bash
#subject 3, database 3
python scripts/load_results.py 4 database_3
#subject 14, database 1
python scripts/load_results.py 14 database_1
```
### Ploting results for database 1
```bash
#subject 10, database 1
python scripts/plot_data.py 10
```
## Running training sessions

### Launching tensorboard for visualizing the training process (optional)
```bash
tensorboard --logdir=results/logs/ &
```
### Running training for all datasets in all databases
```bash
python deepconvlstm/session_run.py
```
### Running training for one entire database
```python
python
>>>from deepconvlstm import session_run
>>>session_run.run_dbtraining('database_1')
```

###  Running training for one subject only
```python
python
>>>from deepconvlstm import DeepConvLstm
>>>subject = 10
>>>database = 'database_1'
>>>dcl = DeepConvLstm(subject, database)
>>>dcl.run_training()
```
## Using jupyter notebooks

## Launch jupyter
```bash
jupyter notebook notebooks/
```
