"""Versioned mean-teacher routing; frozen flat/set implementations unchanged."""
from utils.flow_iql_research import onestep_implementation_registry_v1 as previous
Implementation=previous.Implementation
FLAT=previous.FLAT
SET=previous.SET
SET_RECIPES=previous.SET_RECIPES
MEAN_TEACHER=Implementation('mean_teacher','utils.flow_iql_research.mean_teacher_onestep_core_v1',
 'utils.flow_iql_research.train_mean_teacher_onestep_v1',
 'utils.flow_iql_research.evaluate_mean_teacher_onestep_v1',
 'utils.flow_iql_research.benchmark_mean_teacher_onestep_v1')
IMPLEMENTATIONS=(*previous.IMPLEMENTATIONS,MEAN_TEACHER)

def implementation(recipe):
 if '__meanTeacher' in recipe:
  assert recipe in MEAN_TEACHER.core.RECIPES,('Unvalidated mean teacher recipe',recipe)
  return MEAN_TEACHER
 return previous.implementation(recipe)

def specification(recipe):
 selected=implementation(recipe)
 if selected.tag=='mean_teacher':return selected.core.specification(recipe)
 return previous.specification(recipe)

def configuration(environment,recipe,seed,*args,**kwargs):
 return implementation(recipe).runner.configuration(environment,recipe,seed,*args,**kwargs)

def config_for(environment,recipe):return implementation(recipe).core.config_for(environment,recipe)

def training_sources(plan,recipe):return plan['implementation_training_source_sha256'][implementation(recipe).tag]
