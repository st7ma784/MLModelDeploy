
import wandb
from trainscript import train
import argparse
from functools import partial
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", default="SWEEPID",nargs="?", type=str)
parser.add_argument("--data_dir", default=".",nargs="?", type=str) # edit if machines have tmp or spare storage
parser.add_argument("--devices", default="auto",nargs="?", type=str)
parser.add_argument("--accelerator", default="auto",nargs="?", type=str)

p = parser.parse_args()
train=partial(train,dir=p.data_dir,devices=p.devices, accelerator=p.accelerator)
wandb.agent(sweep_id=p.sweep_id, project="WANDBPROJECT", entity="WANDBUSER",function=train)
