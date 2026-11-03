"""Add joint latent selectors; preserve every previous implementation route."""
from utils.flow_iql_research import onestep_implementation_registry_v4 as previous

Implementation=previous.Implementation
FLAT,SET,MEAN_TEACHER,ENERGY,MIXTURE=(previous.FLAT,previous.SET,previous.MEAN_TEACHER,previous.ENERGY,previous.MIXTURE)
SET_RECIPES=previous.SET_RECIPES
LATENT=Implementation('latent','utils.flow_iql_research.latent_selector_onestep_core_v1',
    'utils.flow_iql_research.train_latent_selector_onestep_v1',
    'utils.flow_iql_research.evaluate_latent_selector_onestep_v1',
    'utils.flow_iql_research.benchmark_latent_selector_onestep_v1')
GUIDED=Implementation('latent_guided','utils.flow_iql_research.latent_order_guided_core_v1',
    'utils.flow_iql_research.train_latent_order_guided_v1',
    'utils.flow_iql_research.evaluate_latent_order_guided_v1',
    'utils.flow_iql_research.benchmark_latent_order_guided_v1')
IMPLEMENTATIONS=(*previous.IMPLEMENTATIONS,LATENT,GUIDED)
GUIDED_RECIPES=('latent_order9_q_a003','latent_order9_q_a03','latent_order9_q_a3')
LATENT_RECIPES=('latent_order9','latent_diag9')


def implementation(recipe):
    if recipe in GUIDED_RECIPES:return GUIDED
    if recipe.startswith('latent_'):
        assert recipe in LATENT_RECIPES,('Unsupported latent recipe/suffix',recipe)
        return LATENT
    return previous.implementation(recipe)


def specification(recipe):
    if recipe in GUIDED_RECIPES:return GUIDED.core.specification(recipe)
    if recipe in LATENT_RECIPES:return LATENT.core.specification(recipe)
    return previous.specification(recipe)


def configuration(environment,recipe,seed,*args,**kwargs):
    return implementation(recipe).runner.configuration(environment,recipe,seed,*args,**kwargs)


def config_for(environment,recipe):return implementation(recipe).core.config_for(environment,recipe)


def training_sources(plan,recipe):
    return plan['implementation_training_source_sha256'][implementation(recipe).tag]
