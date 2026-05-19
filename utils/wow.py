import wandb
api = wandb.Api()
run = api.run("/ukjo19/ORL-BIAS/runs/0d11fbfc-0abf-4eb3-a286-5ae6d09092a3")


print(run.history())
