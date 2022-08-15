export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
#Doing this to save using our home disk space ! 
mkdir -p $CONDADIR
pushd $CONDADIR

# Download the latest miniconda installer for ppcle64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
# Validate the file checksum matches is listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
sha256sum Miniconda3-latest-Linux-ppc64le.sh

sh Miniconda3-latest-Linux-ppc64le.sh -b -p ./miniconda
source miniconda/etc/profile.d/conda.sh
conda update conda -y

#on future sessions we'll need to resource this with 
#export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
#source $CONDADIR/miniconda/etc/profile.d/conda.sh
#
