"""Route the experimental energy student; all previous implementations frozen."""
from utils.flow_iql_research import onestep_implementation_registry_v2 as previous

Implementation = previous.Implementation
FLAT,SET = previous.FLAT,previous.SET
SET_RECIPES = previous.SET_RECIPES
MEAN_TEACHER = previous.MEAN_TEACHER
ENERGY = Implementation('energy','utils.flow_iql_research.energy_onestep_deploy_core_v1',
    'utils.flow_iql_research.train_energy_onestep_v1',
    'utils.flow_iql_research.evaluate_energy_onestep_v1',
    'utils.flow_iql_research.benchmark_energy_onestep_v1')
IMPLEMENTATIONS = (*previous.IMPLEMENTATIONS,ENERGY)


def implementation(recipe):
    if recipe == 'energy_bc':
        return ENERGY
    return previous.implementation(recipe)


def specification(recipe):
    if recipe == 'energy_bc':
        return ENERGY.core.specification(recipe)
    return previous.specification(recipe)


def configuration(environment,recipe,seed,*args,**kwargs):
    return implementation(recipe).runner.configuration(environment,recipe,seed,*args,**kwargs)


def config_for(environment,recipe):
    return implementation(recipe).core.config_for(environment,recipe)


def training_sources(plan,recipe):
    return plan['implementation_training_source_sha256'][implementation(recipe).tag]
