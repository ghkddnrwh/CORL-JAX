from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool


class TwoPathBranchingMDP:
    """Deterministic two-path MDP.

    States:
      s_0,
      s_1_1, ..., s_1_{N-1}, s_1_N (= terminal),
      s_2_1, ..., s_2_{2N-1}, s_2_{2N} (= same terminal).

    From s_0 there are two valid actions:
      action 0 -> short branch
      action 1 -> long branch

    On both branches there are A valid actions, all transition-identical.
    Taking any valid branch action advances one step forward along the chosen branch.

    The reward is 0 everywhere except on the transition into the shared terminal,
    where reward_value is received and the episode terminates.

    Notes:
      - The overall tabular action dimension is max(2, A).
      - Action semantics are state-dependent: action 0/1 at the root chooses the branch,
        while on branch states actions 0..A-1 are equivalent transition-wise.
      - The short branch is uniquely optimal only when gamma < 1.
    """

    def __init__(self, short_horizon: int, branch_num_actions: int, reward_value: float = 1.0):
        if short_horizon <= 0:
            raise ValueError("short_horizon must be positive.")
        if branch_num_actions <= 0:
            raise ValueError("branch_num_actions must be positive.")

        self.short_horizon = int(short_horizon)
        self.branch_num_actions = int(branch_num_actions)
        self.reward_value = float(reward_value)

        # Global tabular action dimension.
        self.num_actions = max(2, self.branch_num_actions)

        # State indexing.
        # 0: root
        # 1 .. N-1: short internal states s_1_1 .. s_1_{N-1}
        # N .. 3N-2: long internal states s_2_1 .. s_2_{2N-1}
        # 3N-1: shared terminal s_1_N = s_2_{2N}
        self.root_state = 0
        self.short_state_start = 1
        self.short_internal_count = self.short_horizon - 1
        self.long_state_start = self.short_horizon
        self.long_internal_count = 2 * self.short_horizon - 1
        self.terminal_state = 3 * self.short_horizon - 1
        self.num_states = self.terminal_state + 1
        self.initial_state = self.root_state

        self.state_names = self._build_state_names()
        self.action_names = self._build_action_names()

    def _build_state_names(self) -> Dict[int, str]:
        names: Dict[int, str] = {self.root_state: "s_0"}
        for i in range(1, self.short_horizon):
            names[self.short_state_start + (i - 1)] = f"s_1_{i}"
        for i in range(1, 2 * self.short_horizon):
            names[self.long_state_start + (i - 1)] = f"s_2_{i}"
        names[self.terminal_state] = f"s_1_{self.short_horizon}(=s_2_{2 * self.short_horizon})"
        return names

    def _build_action_names(self) -> Dict[int, str]:
        names = {0: "go_short", 1: "go_long"}
        for a in range(self.num_actions):
            if a < self.branch_num_actions:
                names[a] = f"branch_action_{a}"
        # Keep root labels explicit for first two actions in a separate field when exporting.
        return names

    def reset(self) -> int:
        return self.initial_state

    def is_terminal(self, state: int) -> bool:
        return state == self.terminal_state

    def valid_actions(self, state: int) -> np.ndarray:
        self._validate_state(state)
        if state == self.root_state:
            return np.asarray([0, 1], dtype=np.int64)
        if self.is_terminal(state):
            return np.asarray([0], dtype=np.int64)
        return np.arange(self.branch_num_actions, dtype=np.int64)

    def reward(self, state: int, action: int) -> float:
        return float(self.transition(state, action).reward)

    def step(self, state: int, action: int) -> StepResult:
        return self.transition(state, action)

    def _next_state_for_valid_transition(self, state: int) -> int:
        if state == self.root_state:
            # action already validated separately; branch chosen by action id.
            raise RuntimeError("Root transition depends on action and must be handled explicitly.")
        if self.is_terminal(state):
            return self.terminal_state
        # Internal states are ordered topologically so every valid action advances by +1,
        # except the last state on each branch, which advances to the shared terminal.
        if state == self.short_state_start + self.short_internal_count - 1 and self.short_internal_count > 0:
            return self.terminal_state
        if state == self.long_state_start + self.long_internal_count - 1:
            return self.terminal_state
        return state + 1

    def rollout(self, policy: Callable[[int], int]) -> Dict[str, List[int | float | bool | str]]:
        states: List[int] = []
        state_names: List[str] = []
        actions: List[int] = []
        rewards: List[float] = []
        next_states: List[int] = []
        next_state_names: List[str] = []
        dones: List[bool] = []

        state = self.reset()
        chosen_branch = "unknown"

        while True:
            action = int(policy(state))
            self._validate_state_action(state, action)

            if state == self.root_state:
                if action == 0:
                    next_state = self.terminal_state if self.short_horizon == 1 else self.short_state_start
                    chosen_branch = "short"
                else:
                    next_state = self.long_state_start
                    chosen_branch = "long"
                reward = 0.0 if next_state != self.terminal_state else self.reward_value
                done = next_state == self.terminal_state
                step_result = StepResult(next_state=next_state, reward=reward, done=done)
            else:
                step_result = self.step(state, action)

            states.append(state)
            state_names.append(self.state_names[state])
            actions.append(action)
            rewards.append(float(step_result.reward))
            next_states.append(step_result.next_state)
            next_state_names.append(self.state_names[step_result.next_state])
            dones.append(step_result.done)
            if step_result.done:
                break
            state = step_result.next_state

        return {
            "states": states,
            "state_names": state_names,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "next_state_names": next_state_names,
            "dones": dones,
            "chosen_branch": [chosen_branch],
        }

    def path_return(self, branch: str, gamma: float) -> float:
        if branch not in {"short", "long"}:
            raise ValueError("branch must be 'short' or 'long'")
        reward_timestep = self.short_horizon - 1 if branch == "short" else (2 * self.short_horizon - 1)
        return float((gamma ** reward_timestep) * self.reward_value)

    def optimal_branch(self, gamma: float) -> str:
        short_return = self.path_return("short", gamma)
        long_return = self.path_return("long", gamma)
        if short_return >= long_return:
            return "short"
        return "long"

    def as_dict(self) -> Dict[str, object]:
        return {
            "env_name": "two_path_branching_mdp",
            "short_horizon": self.short_horizon,
            "branch_num_actions": self.branch_num_actions,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "reward_value": self.reward_value,
            "initial_state": self.initial_state,
            "terminal_state": self.terminal_state,
            "state_names": self.state_names,
            "root_action_meaning": {0: "short_branch", 1: "long_branch"},
            "branch_action_meaning": {
                a: "advance_along_chosen_branch" for a in range(self.branch_num_actions)
            },
        }

    def _validate_state(self, state: int) -> None:
        if not (0 <= state < self.num_states):
            raise IndexError(f"invalid state {state}")

    def _validate_state_action(self, state: int, action: int) -> None:
        self._validate_state(state)
        if not (0 <= action < self.num_actions):
            raise IndexError(f"invalid action {action}")
        valid = self.valid_actions(state)
        if action not in set(valid.tolist()):
            raise ValueError(
                f"invalid action {action} at state {self.state_names[state]}; valid actions are {valid.tolist()}"
            )

    def transition(self, state: int, action: int) -> StepResult:
        """Single-step transition used by both rollout and dynamic programming."""
        self._validate_state_action(state, action)
        if self.is_terminal(state):
            raise ValueError("Cannot step from the terminal state.")
        if state == self.root_state:
            if action == 0:
                next_state = self.terminal_state if self.short_horizon == 1 else self.short_state_start
            else:
                next_state = self.long_state_start
        else:
            next_state = self._next_state_for_valid_transition(state)
        reward = float(self.reward_value if next_state == self.terminal_state else 0.0)
        done = next_state == self.terminal_state
        return StepResult(next_state=next_state, reward=reward, done=done)


def evaluate_policy(
    env: TwoPathBranchingMDP,
    policy: Callable[[int], int],
    gamma: float = 1.0,
) -> Dict[str, object]:
    trajectory = env.rollout(policy)
    discounted_return = 0.0
    for t, reward in enumerate(trajectory["rewards"]):
        discounted_return += (gamma ** t) * float(reward)

    chosen_branch = str(trajectory["chosen_branch"][0])
    optimal_branch = env.optimal_branch(gamma)
    return {
        "discounted_return": float(discounted_return),
        "actions": [int(a) for a in trajectory["actions"]],
        "rewards": [float(r) for r in trajectory["rewards"]],
        "states": [int(s) for s in trajectory["states"]],
        "state_names": [str(x) for x in trajectory["state_names"]],
        "next_states": [int(s) for s in trajectory["next_states"]],
        "next_state_names": [str(x) for x in trajectory["next_state_names"]],
        "chosen_branch": chosen_branch,
        "chose_short_path": bool(chosen_branch == "short"),
        "selected_optimal_path": bool(chosen_branch == optimal_branch),
        "optimal_branch": optimal_branch,
        "short_path_return": float(env.path_return("short", gamma)),
        "long_path_return": float(env.path_return("long", gamma)),
    }


def optimal_policy_and_return(
    env: TwoPathBranchingMDP,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Exact optimal policy and Q-values with invalid actions masked to -inf."""
    q_star = np.full((env.num_states, env.num_actions), -np.inf, dtype=np.float64)
    v_star = np.zeros(env.num_states, dtype=np.float64)

    for state in reversed(range(env.num_states)):
        if env.is_terminal(state):
            valid_actions = env.valid_actions(state)
            q_star[state, valid_actions] = 0.0
            v_star[state] = 0.0
            continue

        valid_actions = env.valid_actions(state)
        for action in valid_actions:
            step_result = env.transition(state, int(action))
            q_star[state, action] = float(step_result.reward + gamma * v_star[step_result.next_state])
        v_star[state] = float(np.max(q_star[state, valid_actions]))

    optimal_policy = np.zeros(env.num_states, dtype=np.int64)
    for state in range(env.num_states):
        valid_actions = env.valid_actions(state)
        optimal_policy[state] = int(valid_actions[np.argmax(q_star[state, valid_actions])])

    return optimal_policy, float(v_star[env.initial_state]), q_star
