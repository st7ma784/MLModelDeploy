# MLModelDeploy
A repo for barebones model development

## Bede Cluster
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


    
    
## Ray + SLURM

Ray uses a head + workers model, once your script works, ideally with PL, look at using the slurm launcher (needs to be called within cluster) (slurm-launch.py) to auto-generate SLURM scripts and launch. slurm-launch.py uses an underlying template (slurm-template.sh) and fills out placeholders given user input.

Usage example
If you want to utilize a multi-node cluster in slurm:

```
python slurm-launch.py --exp-name test --command "python your_file.py" --num-nodes 3

```


If you want to specify the computing node(s), just use the same node name(s) in the same format of the output of sinfo command:

```

python slurm-launch.py --exp-name test --command "python your_file.py" --num-nodes 3 --node NODE_NAMES

```


There are other options you can use when calling python slurm-launch.py:

--exp-name: The experiment name. Will generate {exp-name}_{date}-{time}.sh and {exp-name}_{date}-{time}.log.

--command: The command you wish to run. For example: rllib train XXX or python XXX.py.

--num-gpus: The number of GPUs you wish to use in each computing node. Default: 0.

--node (-w): The specific nodes you wish to use, in the same form as the output of sinfo. Nodes are automatically assigned if not specified.

--num-nodes (-n): The number of nodes you wish to use. Default: 1.

--partition (-p): The partition you wish to use. Default: “”, will use user’s default partition.

--load-env: The command to setup your environment. For example: module load cuda/10.1. Default: “”.

Note that the slurm-template.sh is compatible with both IPV4 and IPV6 ip address of the computing nodes.

### Implementation
Concretely, the (slurm-launch.py) does the following things:

It automatically writes your requirements, e.g. number of CPUs, GPUs per node, the number of nodes and so on, to a sbatch script name {exp-name}_{date}-{time}.sh. Your command (--command) to launch your own job is also written into the sbatch script.

Then it will submit the sbatch script to slurm manager via a new process.

Finally, the python process will terminate itself and leaves a log file named {exp-name}_{date}-{time}.log to record the progress of your submitted command. At the mean time, the ray cluster and your job is running in the slurm cluster.

### Known Networking Bugs:
There are some known issues around multiple users and port number conflicts: 
More details at [Ray+Slurm](https://docs.ray.io/en/master/cluster/slurm.html)

## Cog: Containers for machine learning

Cog is an open-source tool that lets you package machine learning models in a standard, production-ready container.

You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/).

### Highlights

- 📦 **Docker containers without the pain.** Writing your own `Dockerfile` can be a bewildering process. With Cog, you define your environment with a [simple configuration file](#how-it-works) and it generates a Docker image with all the best practices: Nvidia base images, efficient caching of dependencies, installing specific Python versions, sensible environment variable defaults, and so on.

- 🤬️ **No more CUDA hell.** Cog knows which CUDA/cuDNN/PyTorch/Tensorflow/Python combos are compatible and will set it all up correctly for you.

- ✅ **Define the inputs and outputs for your model with standard Python.** Then, Cog generates an OpenAPI schema and validates the inputs and outputs with Pydantic.

- 🎁 **Automatic HTTP prediction server**: Your model's types are used to dynamically generate a RESTful HTTP API using [FastAPI](https://fastapi.tiangolo.com/).

- 🥞 **Automatic queue worker.** Long-running deep learning models or batch processing is best architected with a queue. Cog models do this out of the box. Redis is currently supported, with more in the pipeline.

- ☁️ **Cloud storage.** Files can be read and written directly to Amazon S3 and Google Cloud Storage. (Coming soon.)

- 🚀 **Ready for production.** Deploy your model anywhere that Docker images run. Your own infrastructure, or [Replicate](https://replicate.com).

## How it works

Define the Docker environment your model runs in with `cog.yaml`:

Define how predictions are run on your model with `model.py`:

Now, you can run predictions on this model:

```
$ cog predict -i @input.jpg
--> Building Docker image...
--> Running Prediction...
--> Output written to output.jpg
```

Or, build a Docker image for deployment: (an automated github action on push)

```
$ cog build -t my-colorization-model
$ docker run -d -p 5000:5000 --gpus all my-colorization-model

$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"image": "https://.../input.jpg"}}'
```

In development, you can also run arbitrary commands inside the Docker environment:

```
$ cog run python train.py
...
```

Or, [spin up a Jupyter notebook](docs/notebooks.md):

```
$ cog run -p 8888 jupyter notebook --allow-root --ip=0.0.0.0
```
-->

## Why are we building this?

It's really hard for researchers to ship machine learning models to production.

Part of the solution is Docker, but it is so complex to get it to work: Dockerfiles, pre-/post-processing, Flask servers, CUDA versions. More often than not the researcher has to sit down with an engineer to get the damn thing deployed.

## Prerequisites

- **macOS or Linux**. Cog works on macOS and Linux, but does not currently support Windows.
- **Docker**. Cog uses Docker to create a container for your model. You'll need to [install Docker](https://docs.docker.com/get-docker/) before you can run Cog.

## Install

First, [install Docker if you haven't already](https://docs.docker.com/get-docker/). Then, run this in a terminal:

```
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```
or simply use the script Install.sh

## Upgrade

If you're already got Cog installed and want to update to a newer version:

```
sudo rm $(which cog)
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```
