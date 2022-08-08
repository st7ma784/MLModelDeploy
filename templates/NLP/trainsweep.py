import wandb
wandb.login()
if __name__=="__main__":
    sweep_config = {
        'method': 'bayes',  # Randomly sample the hyperparameter space (alternatives: random, grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
            },
            'batch_size': {
                'values': [1,4,8,12]
            },
            'precision': {
                'values': [32,16,'bf16']
            },
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="PROJECTNAME", entity="WANDB_USER")
    print(sweep_id)
