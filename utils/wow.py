import wandb
api = wandb.Api()
run = api.run("/ukjo19/ORL-SMOOTH/runs/673c7b9e-2443-4fc4-9e8f-dd52e6861697")

print(run.history())
