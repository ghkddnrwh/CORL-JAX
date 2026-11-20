"""Separate a deployment candidate key from its immutable training recipe."""
from dataclasses import replace
from utils.flow_iql_research import onestep_implementation_registry_v5 as native
from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending as native_order

POLICY='mdn9_bc__component_location'
NATIVE='mdn9_bc'
MODE='component_location'
DEPLOYMENT_ID='mdn9_component_location_v1'
SET_RECIPES=native.SET_RECIPES


def training_recipe(recipe):
    return NATIVE if recipe==POLICY else recipe


def implementation(recipe):
    result=native.implementation(training_recipe(recipe))
    if recipe==POLICY:
        return replace(result,
            evaluator_module='utils.flow_iql_research.evaluate_portable_mixture_locations_v1',
            benchmark_module='utils.flow_iql_research.benchmark_routed_mixture_isolated_v1')
    return result


def specification(recipe):
    result=native.specification(training_recipe(recipe))
    if recipe==POLICY:
        result=dict(result,deployment_candidate=POLICY,deployment_rule='sampled_component_location',
                    native_training_recipe=NATIVE)
    return result


def configuration(environment,recipe,seed,*args,**kwargs):
    return native.configuration(environment,training_recipe(recipe),seed,*args,**kwargs)


def config_for(environment,recipe):
    return native.config_for(environment,training_recipe(recipe))


def training_sources(plan,recipe):
    return native.training_sources(plan,training_recipe(recipe))


def order_pending(plan,state,queue,started):
    # Scheduling matches the real learner recipe without changing any job.
    adapted=dict(state)
    if state.get('active_attempt')is not None:
        adapted['attempts']=[dict(x,recipe=training_recipe(x['recipe'])) for x in state['attempts']]
    return native_order(plan,adapted,queue,started)


def preferred_candidate(state,plan,candidates):
    extension=plan.get('routed_deployment_extension')
    if extension and POLICY not in state['tested_candidates']:
        return dict(recipe=POLICY,native_training_recipe=NATIVE,eligible=False,
            selection_kind='registered_common_location_policy_discovery',exploratory_promotion=True,
            deployment_rule='sampled_component_location',
            reason='One globally fixed decoder selected from four same-model seed0 discovery tasks; all nine tasks and three fresh training seeds remain mandatory.',
            no_original_pilot_pass_claim=True,all_final_gates_unchanged=True)
    from utils.flow_iql_research.discovery_bound_policy_v1 import preferred_candidate as previous
    return previous(state,plan,candidates)
