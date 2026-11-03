"""Explicit implementation routing for frozen flat and set actor sources."""
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path


SET_RECIPES=('set_bc','set_q_a03','set_q_a1','set_q_a3','set_value_a1','set_regret_a1')


@dataclass(frozen=True)
class Implementation:
    tag: str
    core_module: str
    runner_module: str
    evaluator_module: str
    benchmark_module: str

    @property
    def core(self):return load(self.core_module)

    @property
    def runner(self):return load(self.runner_module)

    def path(self,module):return Path(__file__).resolve().with_name(module.rsplit('.',1)[1]+'.py')

    @property
    def core_path(self):return self.path(self.core_module)

    @property
    def evaluator_path(self):return self.path(self.evaluator_module)

    @property
    def benchmark_path(self):return self.path(self.benchmark_module)


@lru_cache(None)
def load(module):return importlib.import_module(module)


FLAT=Implementation('flat','utils.flow_iql_research.onestep_deploy_core_v1',
    'utils.flow_iql_research.train_onestep_deploy_v1',
    'utils.flow_iql_research.evaluate_onestep_deploy_v1',
    'utils.flow_iql_research.benchmark_onestep_deploy_v1')
SET=Implementation('set','utils.flow_iql_research.set_onestep_deploy_core_v1',
    'utils.flow_iql_research.train_set_onestep_deploy_v1',
    'utils.flow_iql_research.evaluate_set_onestep_deploy_v1',
    'utils.flow_iql_research.benchmark_set_onestep_deploy_v1')
IMPLEMENTATIONS=(FLAT,SET)


def implementation(recipe):
    if recipe.startswith('set_'):
        assert recipe in SET_RECIPES,('Unregistered set architecture suffix',recipe)
        return SET
    FLAT.core.specification(recipe)
    return FLAT


def specification(recipe):
    selected=implementation(recipe)
    if selected.tag=='flat':return selected.core.specification(recipe)
    return selected.core.config_for(FLAT.core.ENVIRONMENTS[0],recipe)['onestep_specification']


def configuration(environment,recipe,seed,*args,**kwargs):
    return implementation(recipe).runner.configuration(environment,recipe,seed,*args,**kwargs)


def config_for(environment,recipe):
    return implementation(recipe).core.config_for(environment,recipe)


def training_sources(plan,recipe):
    return plan['implementation_training_source_sha256'][implementation(recipe).tag]
