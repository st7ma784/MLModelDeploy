#$ -S /bin/bash
#$ -q short
#$ -l ngpus=1
#$ -l ncpus=6
#$ -l h_vmem=80G
#$ -l h_rt=12:00:00
#$ -M {YOUREMAIL} 
#$ -m beas   # Email on Begin, End, Abortted, stopped
source /etc/profile
module add anaconda3/wmlce   #User built in Watson Machine Learning Conda Env
source activate $global_storage/conda4 #If you have your own env created, activate it here. 
modelcache=$global_storage/data/pretraining  # set an OS Var for pretrained models if your script uses them...

module add git # load git 

### Create Data Dir- in this instance pull from a git repo, may be worth doing each time if using temp storage 
cd $global_scratch 
git clone https://github.com/carlosGarciaHe/MS-COCO-ES.git

### Now move to our code base
cd $global_storage/NDimRL
git pull # check for updates, saves constantly reloading scripts or trying to edit on the HEC itself. 

# Set some flags if useful...

export WANDB_SILENT=true
export WANDB_RESUME=auto
export WANDB_CONSOLE='off'
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
export ISHEC=1 # This can be used if dataloaders cause issues with memory size
# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# Add a Wandb User key so that the run doesnt hang. 
export wandb='40 char User key'

python trainagent.py --data_dir $global_scratch/ms-coco-es --log_dir $TMPDIR
