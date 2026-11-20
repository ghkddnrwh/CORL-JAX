"""Add the mixture actor; route all previous recipes to unchanged sources."""
from utils.flow_iql_research import onestep_implementation_registry_v3 as previous

Implementation=previous.Implementation
FLAT,SET,MEAN_TEACHER,ENERGY=previous.FLAT,previous.SET,previous.MEAN_TEACHER,previous.ENERGY
SET_RECIPES=previous.SET_RECIPES
MIXTURE=Implementation('mixture','utils.flow_iql_research.mixture_onestep_deploy_core_v1',
    'utils.flow_iql_research.train_mixture_onestep_v1',
    'utils.flow_iql_research.evaluate_mixture_onestep_v1',
    'utils.flow_iql_research.benchmark_mixture_onestep_v1')
IMPLEMENTATIONS=(*previous.IMPLEMENTATIONS,MIXTURE)
MIXTURE_RECIPES=('mdn9_bc','mdn9_q_a003','mdn9_q_a03','mdn9_q_a3')


def implementation(recipe):
    if recipe.startswith('mdn9_'):
        assert recipe in MIXTURE_RECIPES,('Unsupported mixture recipe/suffix',recipe)
        return MIXTURE
    return previous.implementation(recipe)


def specification(recipe):
    if recipe in MIXTURE_RECIPES:return MIXTURE.core.specification(recipe)
    return previous.specification(recipe)


def configuration(environment,recipe,seed,*args,**kwargs):
    return implementation(recipe).runner.configuration(environment,recipe,seed,*args,**kwargs)


def config_for(environment,recipe):return implementation(recipe).core.config_for(environment,recipe)


def training_sources(plan,recipe):
    return plan['implementation_training_source_sha256'][implementation(recipe).tag]
