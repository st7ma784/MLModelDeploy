
# Bede Cluster
This repo is designed to upscale as far as the HEC and N8 BEDE CLuster, the latter uses SLURM but has the following directory structures :

#### Project home directory
/projects/<project>
Intended for project files to be backed up (note: backups not currently in place)
Modest performance
A default quota of 20GB

#### Project Lustre directory 
/nobackup/projects/<project>
Intended for bulk project files not requiring backup
Fast performance
No quota limitations
By default, files created within a project area are readable and writable by all other members of that project.

#### Home directory 
   /users/<user>


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
