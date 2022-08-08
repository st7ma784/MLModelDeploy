def train(config={
        "batchsize":16,
        "learning_rate":2e-4,
        "precision":16,
    },dir="/Data",devices="auto",accelerator="auto"):
    #Load Data Module and begin training
    from DataModule import DataModule
    from model import GLUETransformer
    with wandb.init( project="PROJECTNAME", entity="WANDBUSERNAME", job_type="train", config=config) as run:  
        model=GLUETransformer(  learning_rate = config["learning_rate"],
                                    train_batch_size=config["batchsize"],
                                    adam_epsilon = 1e-8)
        Dataset=DataModule(Cache_dir=dir,batch_size=config["batchsize"])
        callbacks=[
            TQDMProgressBar()
        ]
        logtool= pytorch_lightning.loggers.WandbLogger(experiment=run)
        trainer=pytorch_lightning.Trainer(
            devices=devices,
            accelerator=accelerator,
            max_epochs=100,
            logger=logtool,
            callbacks=callbacks,
            gradient_clip_val=0.25,
            precision=config["precision"]
        )
        
        
        trainer.fit(model,Dataset)

if __name__ == '__main__':
    config={
        "batchsize":12,         #[1,4,8,16,32,64]
        "learning_rate":4e-6,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":16,         #[32,16,'bf16']
    }
