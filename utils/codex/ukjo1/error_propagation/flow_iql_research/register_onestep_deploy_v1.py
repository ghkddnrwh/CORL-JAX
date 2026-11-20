"""Preregister actor-only deployment search before production learner outcomes."""
import fcntl
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research import train_onestep_deploy_v1 as runner


SECOND_ROUND=(
    'bank_q_a10','bank_q_a30','bank_value_a03','bank_value_a3',
    'bank_value_a10','bank_regret_a03','bank_regret_a3','bank_regret_a10',
    'bank_bc__w2','bank_bc__ln','bank_bc__d6','bank_bc__current',
    'state_q_a03','state_q_a3','fqln_a30','fqln_a100',
)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('logs/diagnosis/finite_mean_onestep_20261031').resolve();u=Path(__file__).resolve().parent
    assert not (root/'registration_v1.json').exists()
    validation=[]
    for part in (0,1):
        d=json.loads((root/f'analysis/update_validation_part{part}_v3.json').read_text());assert d['status']=='passed'
        for path,digest in d['source_sha256'].items():assert sha(path)==digest,path
        validation+=d['cases']
    assert {r['recipe']for r in validation}==set(core.INITIAL_RECIPES)
    smoke=json.loads((root/'analysis/runner_smoke_validation_v1.json').read_text());assert smoke['status']=='passed'
    protected=u/'FINITE_MEAN_REFERENCE_MANIFEST_V1.json';protection=json.loads(protected.read_text())
    for path,digest in protection['frozen_original_source_sha256'].items():assert sha(path)==digest,path
    pilots=[core.ENVIRONMENTS[0],core.ENVIRONMENTS[5],core.ENVIRONMENTS[1]]
    jobs=[];identities={}
    for environments,recipes in [(pilots[:2],core.INITIAL_RECIPES[:8]),(pilots[:2],core.INITIAL_RECIPES[8:]),(pilots[2:],core.INITIAL_RECIPES)]:
        for recipe in recipes:
            for environment in environments:
                tag='ant' if environment==pilots[0] else 'cubed' if environment==pilots[1] else 'hmmed'
                rid=f'os1_{tag}_{recipe}_s0'
                job=dict(id=rid,module='utils.flow_iql_research.train_onestep_deploy_v1',args=dict(env=environment,recipe=recipe,seed=0,steps=1000000))
                jobs.append(job);identities[rid]=runner.configuration(environment,recipe,0)
    assert len(jobs)==48 and len({j['id']for j in jobs})==48
    training={str(path):sha(path)for path in core.source_paths()+[Path(runner.__file__).resolve()]}
    all_sources=training|{str(u/name):sha(u/name)for name in ('evaluate_onestep_deploy_v1.py','benchmark_onestep_deploy_v1.py','validate_onestep_deploy_v1.py','scheduler_onestep_deploy_v1.py','scheduler_v4.py','register_onestep_deploy_v1.py')}
    for recipe in SECOND_ROUND:core.config_for(pilots[0],recipe)
    result=dict(time=time.time(),root=str(root),workspace_link=str(Path('logs/diagnosis/finite_mean_onestep_20261031').absolute()),
        requested_goal='Leanfinite-meanwithjointactortraining,actor-onlysingle-forwarddeployment,retainorimprove9taskperformance. Continueadaptiveexperimentsuntilachieved.',
        protected_original_manifest=str(protected),protected_original_manifest_sha256=sha(protected),
        jobs=jobs,expected_identities=identities,training_source_sha256=training,all_source_sha256=all_sources,
        initial_recipes=list(core.INITIAL_RECIPES),initial_recipe_specs={name:core.specification(name)for name in core.INITIAL_RECIPES},
        pilot_environments=pilots,initial_training_seed=0,primary_steps=[800000,900000,1000000],episodes=50,first_episode_seed=42,
        auxiliary_steps=[100000,300000],auxiliary_episodes=10,
        metric='Equal CPU50success at800K/900K/1M;equaltasks;equaltrainingseeds. Separate actorprimary andsamecheckpointteacherbaseline. Nopeak/missingzeros/taskwisewinners.',
        CPUslots=[list(range(48+6*i,54+6*i))for i in range(8)],GPU_slots_per_device=8,memory_fraction='.1',
        pilot_reference={pilots[0]:.46,pilots[1]:.32,pilots[2]:.82},
        thresholds=dict(mean_loss_tolerance=.03,max_task_loss=.10,CPU_latency_ratio_max=.5,GPU_latency_ratio_max=2/3,
            historical_lean_seed0_nine_task_reference=.6251851851851852,historical_original_fresh_mean_reference=.5888888888888889,
            pilot_historical_mean_reference=.5333333333333333,
            scope='Practical descriptive performance tolerances, not a formal statistical noninferiority test.'),
        selection='Comparecompleted3taskseed0recipes. Requirepairedteachermeanloss<=.03,maxindividualteacherandhistoricaltaskloss<=.10,pilotprimarymean>=historicalminus.03. Choosehighestprimarymean;within.01preferfeweractorparams/noguidance/simplerrecipe. Oneglobalrecipe.',
        confirmation='Atmostoneactivecandidate;registermissing6seed0tasksplus9tasksx3freshseedsbeforetraining. Freshseeds12+3*attempt,13+3*attempt,14+3*attempt. Pairedsamecheckpointteacherbaseline forEVERYrun. Testfull9seed0andfreshmacro againstpairedteacherandhistoricalthresholds;thenpairedisolatedCPU/GPUlatencyover9seed0models. Failedcandidate=>nexteligibleornewround,notSTOP.',
        adaptive_search=dict(second_round=list(SECOND_ROUND),later='Centerglobalalpha/width/depth/layernorm/current-statevariantsaroundbestfamilies. Registereachnewbatchbeforeoutcomes;nonewsourcehotedits. Ifnewarchitecture/lossneeded,writerequestandcontinueexplicitactivegoalwithversionedsource.',
            failure_rule='Nofirst-roundjointsuccessdoesnotendthisresearchgoal. Preservefailures,inspectteacher/actorregret/saturation/diversity/rootmetrics andcontinue.'),
        tuning_sources={str(Path('algorithms/uk_offline/basic/propagation/fql/README.md').resolve()):sha('algorithms/uk_offline/basic/propagation/fql/README.md'),str(Path('algorithms/uk_offline/basic/propagation/fql.pdf').resolve()):sha('algorithms/uk_offline/basic/propagation/fql.pdf')},
        storage='AllnewmodelsliveontheHDDroot fromthestart. Originalsuccessfulsources/models/resultsunchanged.',
        implementation_validation=[str(root/f'analysis/update_validation_part{p}_v3.json')for p in (0,1)],runner_smoke=str(root/'analysis/runner_smoke_validation_v1.json'),
        scientific_notes=['FQLnormalizedalpha [.03,.1,.3,1,3,10]comesfromofficialREADMErecommendation;rawpublicalpha multipliersaregloballyshared.',
            'Best9labelsuseoldONLINEmin-Qselection,whilebackupremainsoldTARGETmean-Qfiniteexpectile.',
            'Bankactorconsumes9GaussianlatentvectorsandoutputsONEactionwithONEMLPforward. Itdoesnotproduce9candidateactionsorruncritic/flow atdeployment.',
            'AlldeploymentactorshaveindependentAdamandneverinfluencebootstrap/critic/BCgradients. Mathematicalbaselearnerisfixed;long-runfloatingcompilationsmaydiffer,soeveryrunhasmatchedteachercontrol.',
            'EarlyCPU10screeningisauxiliaryonly. GoaljudgmentusesalllastthreeCPU50evaluationsandfreshtrainingseeds.'])
    (root/'registration_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (root/'queue_mutation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert not (root/'queue.json').exists()
        temp=root/'queue.json.tmp';temp.write_text(json.dumps(jobs,indent=2));temp.replace(root/'queue.json')
    print(json.dumps(dict(root=str(root),jobs=len(jobs),registration_sha256=sha(root/'registration_v1.json'),first16=[j['id']for j in jobs[:16]]),indent=2))


if __name__=='__main__':main()
