from __future__ import annotations

from typing import Any, Dict

from ..env import TwoPathBranchingMDP
from .base import BaseOfflineAlgorithm
from .monte_carlo import MonteCarloConfig, MonteCarloReturnRegressor
from .q_learning import ClippedDoubleQConfig, ClippedDoubleQEnsemble
from .frozen_ratio_value_learning import FrozenRatioValueLearningConfig, FrozenRatioValueLearning

ALGORITHM_REGISTRY = {
    "clipped_q_learning": ClippedDoubleQEnsemble,
    "monte_carlo": MonteCarloReturnRegressor,
    "frozen_ratio_value_learning": FrozenRatioValueLearning,
}

CONFIG_REGISTRY = {
    "clipped_q_learning": ClippedDoubleQConfig,
    "monte_carlo": MonteCarloConfig,
    "frozen_ratio_value_learning": FrozenRatioValueLearningConfig,
}


def build_algorithm(algo_name: str, env: TwoPathBranchingMDP, **kwargs: Any) -> BaseOfflineAlgorithm:
    if algo_name not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm: {algo_name}. Available: {list(ALGORITHM_REGISTRY)}")
    config_cls = CONFIG_REGISTRY[algo_name]
    config = config_cls(**kwargs)
    algo_cls = ALGORITHM_REGISTRY[algo_name]
    return algo_cls(env=env, config=config)


__all__ = [
    "BaseOfflineAlgorithm",
    "ClippedDoubleQConfig",
    "MonteCarloConfig",
    "ClippedDoubleQEnsemble",
    "MonteCarloReturnRegressor",
    "FrozenRatioValueLearningConfig",
    "FrozenRatioValueLearning",
    "build_algorithm",
    "ALGORITHM_REGISTRY",
]
