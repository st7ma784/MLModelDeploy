# MLModelDeploy
A repo for barebones model development

# Cog: Containers for machine learning

Cog is an open-source tool that lets you package machine learning models in a standard, production-ready container.

You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/).

## Highlights

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
