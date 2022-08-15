#BEDE 

The Default Bede Directory structure is : 


## Software 

### Conda 

The install requirements script will INSTALL conda. 

We can build an environment from a file in our main repo with a script like the installEnv.sbatch that under the hood calls the following 

```
export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
source $CONDADIR/miniconda/etc/profile.d/conda.sh
##Activates conda installation 

conda install -f ./environment.yml

```

In the environment.yml file, can either be a list of pip packages OR a -r link to a file path. 

## Data 
 
I would suggest that a good 1 time script to create per use would be to create a dataset pull script, There are merits to using both a 1 time download initially,
and then a conditional pull at runtime (see pytorch-lightning datamodule best practice)  
