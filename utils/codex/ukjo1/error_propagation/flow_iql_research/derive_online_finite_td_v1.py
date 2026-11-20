"""Reconstruct correct online two-head TD RMS without modifying training logs.

For these three profiles critic_loss=mean_batch(sum_heads(error**2)). The
runner averages that loss over every update in a fixed-size window; therefore
sqrt(window_critic_loss/2) is its both-head/batch/update RMS. It is a changing-
state replay-window quantity, not a frozen-data or population Bellman residual.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


PROFILES={'bounded_v2_empirical_backup','bounded_v2_mean_empirical_backup','bounded_v2_anchored_empirical_backup'}


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'manifest.json').exists(),'Use a new version for a later online-log snapshot'
    (out/'raw_prefixes').mkdir(exist_ok=True);(out/'derived').mkdir(exist_ok=True);rows=[];cache={}
    for run in sorted((args.root/'runs').iterdir()):
        if not(run/'config.json').exists():continue
        config=json.loads((run/'config.json').read_text());identity=config['identity']
        if identity['profile']not in PROFILES:continue
        assert identity['kind']=='ifql'and identity['online_metrics']=='all_updates_in_each_probe_window_v2'
        for path,digest in config['source_sha256'].items():
            if path not in cache:cache[path]=sha(Path(path).read_bytes())
            assert cache[path]==digest,path
        path=run/'diagnostics.jsonl'
        if not path.exists():continue
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
        records=[json.loads(line)for line in raw.splitlines()]
        if not records:continue
        derived=[]
        for record in records:
            loss=record['critic/critic_loss'];assert math.isfinite(loss)and loss>=0
            assert record['metric_window_updates']>0
            derived.append(dict(step=record['step'],metric_window_updates=record['metric_window_updates'],
                online_window_projected_actual_TD_RMS=math.sqrt(loss/2),online_window_actual_critic_loss=loss,
                legacy_learned_V_counterfactual_snapshot_td_rms=record['td_rms'],
                auxiliary_learned_V_score_rms=record['score_rms'],snapshot_q_mean=record['q_mean'],snapshot_q_rms=record['q_rms'],
                online_lower_projection_fraction=record.get('backup/lower_clip_fraction'),
                online_upper_projection_fraction=record.get('backup/upper_clip_fraction'),
                online_finite_proposal_score_abs_mean=record.get('backup/finite_proposal_score_abs_mean')))
        rawpath=out/'raw_prefixes'/(run.name+'.jsonl');rawpath.write_bytes(raw)
        target=out/'derived'/(run.name+'.jsonl');target.write_text(''.join(json.dumps(row,allow_nan=False)+'\n'for row in derived))
        rows.append(dict(run=run.name,environment=identity['environment'],profile=identity['profile'],
            records=len(derived),last_step=derived[-1]['step'],last_derived=derived[-1],
            original_log_path=str(path.resolve()),immutable_raw_prefix_sha256=sha(raw),raw_prefix_bytes=len(raw),
            derived_sha256=sha(target.read_bytes()),config_sha256=sha((run/'config.json').read_bytes()),
            parent_complete=(run/'completed.json').exists()))
    result=dict(time=time.time(),kind='corrected_online_finite_target_TD_log_snapshot',runs=rows,
        run_count=len(rows),records=sum(row['records']for row in rows),
        formula='sqrt(online_window_actual_critic_loss/2);twoheads,equalbatchsizes,allupdatesinthewindow',
        original_sources_and_training_logs_unchanged=True,source_sha256=sha(Path(__file__).read_bytes()),
        semantic_note='DIAGNOSTIC_TARGET_SEMANTICS.txt',
        limits=['OnlinewindowsincludechangingQ/targetQ/BCandactualreplaybatches;donotcompareasthesamequantityasfixed-snapshotTD.',
            'Reconstructioninheritsloggedfloat32window-averagingprecision;notoracleconditionalBellmanerrorortrueQerror.',
            'Boundary-auxiliarylossprofileisexcludedbecauseitscritic_lossincludesanextraQboundaryterm.',
            'Incompleteappendtailwasexcluded;immutableprefixcopiespreservethispoint-in-timeevidence.'],goal_complete=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(runs=len(rows),records=result['records'],training_logs_modified=False)),flush=True)


if __name__=='__main__':main()
