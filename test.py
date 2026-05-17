import d4rl
import gym



def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        print(max_ret)
        print(min_ret)
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


env_name = "halfcheetah-medium-v2"

env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(env)
print(dataset["observations"].shape[0])
# 또는
print(len(dataset["terminals"]))


print(modify_reward(dataset=dataset, env_name=env_name))



# import wandb
# api = wandb.Api()
# run = api.run("/ukjo19/ORL-BIAS/runs/fe7345fa-b0f9-4d8e-a77d-4050c16c089c")

# print(run.history())
