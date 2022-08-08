# Here we see the process of changing code from notebook form to PL script

## Step One
So We're in a jupyter notebook, and we can refactor as envs are persistent between splits

This means we can do some basic refactoring. 

## Step Two 

in this step we attack the logic of our training loop
we've tidied a few things up, like instead of manual splits, using the Distributed Samplers

Whilst the caching of runs is reasonably efficient re: parameters,
 it's not very efficient re: data storage/memory and this will be messy on bigger datasets

Therefore we've removed some of the lists of parameters, opting for duplicate models - which PL can automatically save with checkpointing callbacks

## Step Three 

We've split up the encoder model for clarity when defining it, 
but have left in a combined model for legacy support.

The next steps are to look at better approaches with forward hooks. 

We can also reconsider how we're implementing our distributed system ->
At the moment, it's a bit of a hack, but it's a good start. 
We use a round robin between different loaders and sets of parameters 
(we could reload params as per v1, but models is nicer for readability)

## Step Four 
Now we start building the pytorch_lightning script.

Fundamentally, this alllows us to bulk up the training loop to many more users in parralell.
BUT this also removes the need to keep writing ".to(device)" for every single model.

Note that the "model" is a pytorch_lightning.LightningModule, which is a subclass of torch.nn.Module.
We're also doing nothing about sampler(handled by pytorch_lightning) and dataloader(handled by pytorch_lightning).

We also don't worry too much about the loss or when optimizers are created or run (handled by pytorch_lightning).
