# Create the virtual environment
virtualenv -p python3 emg_mc_venv; cd emg_mc_venv
# Activate the environment
source bin/activate; cd ..
# Install the requirements
pip install --no-cache-dir --cert /etc/ssl/certs/ca-bundle.crt -r requirements.txt
# Install deepconvlstm package
python setup.py install
# Install a new kernel
python -m ipykernel install --user --name=emg_env_kernel
# Workaround if the command above exit with permission error
sudo emg_mc_venv/bin/python -m ipykernel install --user --name=emg_env_kernel
