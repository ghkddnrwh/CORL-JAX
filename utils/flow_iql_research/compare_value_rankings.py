"""Separate state offsets from action-ranking changes between value controllers.

This measures differences between two estimates, not either estimate's true bias.
The behavior proposals and heldout query states are exactly shared.
"""
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core, policy_core, selection_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    runs = [root/'runs/w1_scene_official_g999_t90_s0',
            root/'runs/p1_scene_ifql_c0_bounded_v2_g999_t90_s0']
    _, _, validation = ogbench.make_env_and_datasets('scene-play-singletask-v0')
    indices = np.random.RandomState(77231).randint(len(validation['actions']), size=256)
    observations = jnp.asarray(validation['observations'][indices], jnp.float32)
    actions = jnp.asarray(validation['actions'][indices], jnp.float32)
    states = []; sources = {}
    for run, module, kind in zip(runs, (core, policy_core), ('iql', 'ifql')):
        path = run/'checkpoint_1000000.pkl'
        conf = json.loads((run/'config.json').read_text())
        for source, digest in conf['source_sha256'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
        cfg = dict(conf['identity']['config'])
        for field in ('value_hidden_dims', 'actor_hidden_dims'):
            cfg[field] = tuple(cfg[field])
        template = module.create(0, observations[:1], actions[:1], cfg, kind)
        with path.open('rb') as file:
            saved = pickle.load(file)
        assert saved['step'] == 1000000
        states.append(flax.serialization.from_state_dict(template, saved['state']))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    native, corrected = states

    @jax.jit
    def evaluate(obs):
        proposals, qc = selection_v2.proposals(corrected.agent, obs, jax.random.PRNGKey(77128), 32)
        expanded = jnp.broadcast_to(obs, (32, *obs.shape))
        qn = native.agent.network.select('critic')(expanded, actions=proposals).min(0)
        vn = native.agent.network.select('value')(obs)
        vc = corrected.agent.network.select('value')(obs)
        return proposals, qn, qc, vn, vc

    _, qn, qc, vn, vc = [np.asarray(x, np.float64) for x in jax.device_get(evaluate(observations))]
    delta = qc-qn; offset = delta.mean(0); centered = delta-offset
    energy = float(np.mean(delta**2)); offset_energy = float(np.mean(offset**2))
    centered_energy = float(np.mean(centered**2))
    assert abs(energy-offset_energy-centered_energy) < 1e-9*max(energy, 1.)
    rankn = np.argsort(np.argsort(qn, axis=0), axis=0)
    rankc = np.argsort(np.argsort(qc, axis=0), axis=0)
    spearman = 1-6*np.sum((rankn-rankc)**2, axis=0)/(32*(32**2-1))
    bestn = np.argmax(qn, axis=0); bestc = np.argmax(qc, axis=0)
    idx = np.arange(qn.shape[1])
    record = dict(time=time.time(), kind='shared_proposal_controller_difference_decomposition',
        environment='scene-play-singletask-v0', step=1000000, queries=256, proposals_per_state=32,
        summary=dict(controller_Q_difference_RMS=float(np.sqrt(energy)),
            state_mean_shift_RMS=float(np.sqrt(offset_energy)),
            action_centered_Q_difference_RMS=float(np.sqrt(centered_energy)),
            action_centered_fraction_of_difference_energy=centered_energy/max(energy, 1e-30),
            argmax_disagreement_fraction=float(np.mean(bestn != bestc)),
            mean_action_rank_Spearman=float(spearman.mean()),
            native_selected_Q_loss_under_corrected_model=float(np.mean(qc[bestc, idx]-qc[bestn, idx])),
            corrected_selected_Q_loss_under_native_model=float(np.mean(qn[bestn, idx]-qn[bestc, idx])),
            V_difference_RMS=float(np.sqrt(np.mean((vc-vn)**2))),
            action_mean_advantage_difference_RMS=float(np.sqrt(np.mean((offset-(vc-vn))**2)))),
        input_checkpoint_sha256=sources,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        proposal_source_sha256=hashlib.sha256(Path(selection_v2.__file__).read_bytes()).hexdigest(),
        query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        limits=['This decomposes differences between two learned critics, not their errors relative to true Q.',
                'Heldout dataset queries and common generated proposals are not policy-visited states.',
                'State-constant Q shifts preserve action ranks; the decomposition quantifies how much the actual controller change resembles such a shift.',
                'No inference of a performance guarantee, true residual reduction or unique cause from this diagnostic alone.'])
    path = root/'analysis/scene_controller_ranking_decomposition_1M.json'
    assert not path.exists(); tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2)); tmp.replace(path)
    print(json.dumps(record['summary'], indent=2))


if __name__ == '__main__':
    main()
