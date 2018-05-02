# Create the virtual environment
virtualenv -p python3 emg_mc_venv;cd emg_mc_venv
# Clone the repository
git clone https://github.com/jonDel/emg_mc.git
# Install the requirements
pip install -r ../requirements.txt
# Install a new kernel
python -m ipykernel install --user --name=emg_env_kernel
# Workaround if the command above exit with permission error
sudo ./bin/python -m ipykernel install --user --name=emg_env_kernel
# Launch the notebook
jupyter notebook emg_mc/capstone.ipynb
