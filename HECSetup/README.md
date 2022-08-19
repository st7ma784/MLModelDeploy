# HEC 

By default the HEC has a few different storage spaces for usage, 
Location   Size     Backups?     retention
home	10G	Nightly	Permanent
storage	100G	None	Permanent
scratch	10T	None	Files automatically deleted after 4 weeks
temp	Unlimited	None	Files automatically deleted at the end of the job 

Note : temp is only available within BATCH jobs, not from qlogins. 


###useful commands
qgputop - shows use of current runs' GPUs,
qstat - the status of the job queue 
qsub - submit job
touch - update altered time for file - useful for $global_scratch 
unzip -DD  - modify date on unzip implicitly. 



## Software :

### Conda 

### PIP 

## Datasets:

