from test_tube.hpc import  Experiment, HyperOptArgumentParser, SlurmCluster

import torch

def train(hparams, *args):
    """Train your awesome model.
    :param hparams: The arguments to run the model with.
    """
    # Initialize experiments and track all the hyperparameters

    # Save exp when /or rely on auto logging...
    import train from myfile
    train(hparams)


if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log_path', default='$TMPDIR')#We'll assume HEC, 
    parser.opt_list('--y_val',
        default=12, options=[1, 2, 3, 4, 5, 6], tunable=True)
    parser.opt_list('--x_val',
        default=12, options=[20, 12, 30, 45], tunable=True)
    hyperparams = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python3',
        test_tube_exp_name=hyperparams.test_tube_exp_name
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='some@email.com', on_done=True, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules([
        'python-3',
        'anaconda3'
    ])

    # Add commands to the non-SLURM portion.
    
    cluster.add_command('source activate open-ce') # We'll assume that on the BEDE/HEC cluster you've named you conda env after the standard...

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    # cluster.add_slurm_cmd(
    #    cmd='cpus-per-task', value='1', comment='CPUS per task.')

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 4
    cluster.per_experiment_nb_nodes = 2
    #cluster.gpu_type = '1080ti'

    # we'll request 40GB of memory per node
    cluster.memory_mb_per_node = 40000

    # set a walltime of 24 hours,0, minues
    cluster.job_time = '24:00:00'

    # 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
    # you must provide your own loading and saving function which the cluster object will call
    cluster.minutes_to_checkpoint_before_walltime = 1

    # run the models on the cluster
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch') # Change this to optimize_parralel_cluster_cpu to debug.
