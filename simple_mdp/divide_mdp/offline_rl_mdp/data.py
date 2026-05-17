from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np

from .env import TwoPathBranchingMDP


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    trajectory_id: int
    timestep: int


@dataclass
class Trajectory:
    trajectory_id: int
    branch: str
    branch_action: int
    states: List[int]
    actions: List[int]
    rewards: List[float]
    next_states: List[int]
    dones: List[bool]

    def discounted_return(self, gamma: float) -> float:
        return float(sum((gamma ** t) * r for t, r in enumerate(self.rewards)))


class ReplayBuffer:
    def __init__(self, transitions: List[Transition], trajectories: List[Trajectory]):
        self.transitions = transitions
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.transitions)

    def iter_transitions(self) -> Iterable[Transition]:
        return iter(self.transitions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_transitions": len(self.transitions),
            "num_trajectories": len(self.trajectories),
            "transitions": [asdict(t) for t in self.transitions],
            "trajectories": [asdict(traj) for traj in self.trajectories],
        }

    def monte_carlo_targets(self, gamma: float) -> np.ndarray:
        targets = np.zeros(len(self.transitions), dtype=np.float64)
        transition_index = {(tr.trajectory_id, tr.timestep): idx for idx, tr in enumerate(self.transitions)}
        for traj in self.trajectories:
            g = 0.0
            for t in reversed(range(len(traj.rewards))):
                g = float(traj.rewards[t]) + gamma * g
                global_index = transition_index[(traj.trajectory_id, t)]
                targets[global_index] = g
        return targets


def build_branch_action_trajectory_buffer(env: TwoPathBranchingMDP) -> ReplayBuffer:
    """Builds the requested offline dataset with 2A trajectories.

    For each branch action a in {0, ..., A-1}:
      - one trajectory goes root -> short branch, then repeatedly takes branch action a;
      - one trajectory goes root -> long branch, then repeatedly takes branch action a.

    This yields 2A trajectories total.
    """
    all_trajectories: List[Trajectory] = []
    all_transitions: List[Transition] = []

    trajectory_id = 0
    for branch_name, root_action in [("short", 0), ("long", 1)]:
        for branch_action in range(env.branch_num_actions):
            def _policy(state: int, ra=root_action, ba=branch_action) -> int:
                if state == env.root_state:
                    return ra
                if env.is_terminal(state):
                    return 0
                return ba

            rollout = env.rollout(_policy)
            trajectory = Trajectory(
                trajectory_id=trajectory_id,
                branch=branch_name,
                branch_action=branch_action,
                states=[int(x) for x in rollout["states"]],
                actions=[int(x) for x in rollout["actions"]],
                rewards=[float(x) for x in rollout["rewards"]],
                next_states=[int(x) for x in rollout["next_states"]],
                dones=[bool(x) for x in rollout["dones"]],
            )
            all_trajectories.append(trajectory)

            for timestep, (state, action, reward, next_state, done) in enumerate(
                zip(
                    trajectory.states,
                    trajectory.actions,
                    trajectory.rewards,
                    trajectory.next_states,
                    trajectory.dones,
                )
            ):
                all_transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        trajectory_id=trajectory_id,
                        timestep=timestep,
                    )
                )
            trajectory_id += 1

    return ReplayBuffer(transitions=all_transitions, trajectories=all_trajectories)


def buffer_state_action_coverage(buffer: ReplayBuffer, num_states: int, num_actions: int) -> np.ndarray:
    coverage = np.zeros((num_states, num_actions), dtype=np.int64)
    for tr in buffer.transitions:
        coverage[tr.state, tr.action] += 1
    return coverage


def format_action_table(env: TwoPathBranchingMDP) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for state in range(env.num_states):
        valid_actions = set(env.valid_actions(state).tolist())
        row: Dict[str, object] = {
            "state": state,
            "state_name": env.state_names[state],
            "valid_actions": sorted(valid_actions),
        }
        for action in range(env.num_actions):
            if action in valid_actions and not env.is_terminal(state):
                row[f"action_{action}_reward"] = float(env.transition(state, action).reward)
            elif action in valid_actions:
                row[f"action_{action}_reward"] = 0.0
            else:
                row[f"action_{action}_reward"] = None
        rows.append(row)
    return rows
