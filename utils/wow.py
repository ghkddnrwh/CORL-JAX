import wandb
import pandas as pd

import wandb
api = wandb.Api()
run = api.run("/ukjo19/ORL-SMOOTH/runs/b9ce3fbd-bb94-4226-9ffa-5224774d7eb2")

history = run.history()
df = pd.DataFrame(history)
df.to_csv("logs/wandb_logs/relocate_human.csv", index=False)