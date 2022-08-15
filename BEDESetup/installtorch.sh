# Create a new conda environment named torch within your conda installation
conda create -y --name torch python=3.9

# Activate the conda environment
conda activate torch

# Add the OSU Open-CE conda channel to the current environment config
conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

# Also use strict channel priority
conda config --env --set channel_priority strict

# Install the latest available version of PyTorch
conda install -y pytorch

