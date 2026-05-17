# Codex CDAF Variants

These variants keep the original CDAF premise:

`Q(s,a) <- r + gamma V(s')`, then `V(s)` is fitted only as aggressively as the
dataset support justifies.

The common rule is:

- sparse or singleton states keep the dataset action with weight near 1;
- dense states trust local action comparison more;
- dense bad actions are downweighted, not blindly deleted, unless a variant asks
  for a hard gate.

## Implemented Variants

1. `codex_01_support_floor_cdaf.py`
   Dataset kNN support plus a nonzero dense bad-action floor. This is the direct
   "do not erase singleton states" baseline.

2. `codex_02_local_rank_cdaf.py`
   Replaces the absolute `A(s,a) < 0` rule with local delayed-advantage ranking
   among kNN states, so filtering means "bad relative to nearby data".

3. `codex_03_expectile_support_cdaf.py`
   Uses IQL-style expectile value regression. Sparse states use tau near 0.5
   (SARSA/MSE), dense states move toward optimistic expectiles.

4. `codex_04_rank_expectile_cdaf.py`
   Combines local-rank action filtering with support-dependent expectile value
   regression.

5. `codex_05_mild_cql_cdaf.py`
   Adds a small dense-state CQL-style critic penalty using mismatched in-batch
   actions as cheap OOD actions.

6. `codex_06_adaptive_margin_cdaf.py`
   Makes the negative-advantage margin proportional to local advantage spread,
   reducing false rejection when nearby samples disagree.

7. `codex_07_pessimistic_target_cdaf.py`
   Fits V to `min(Q_target, Q_delayed)` and adds a small critic penalty, giving a
   BEAR/CQL/EDAC-like pessimistic bias without adding an ensemble.

8. `codex_08_td_error_trust_cdaf.py`
   Suppresses filtering when the current Bellman residual is large. Bad-action
   labels are trusted only after the critic is locally stable.

9. `codex_09_weighted_actor_cdaf.py`
   Aligns critic filtering with policy extraction by using local-rank CDAF plus
   TD3 weighted behavior cloning.

10. `codex_10_hybrid_support_cdaf.py`
    Uses the max of global dataset kNN support and batch-local support, then
    combines local-rank filtering, TD-error trust, and a small critic penalty.

Run one variant as, for example:

```bash
python algorithms/uk_offline/codex/codex_02_local_rank_cdaf.py \
  --env antmaze-large-play-v2 \
  --checkpoints_path logs/tuning/codex_02_local_rank_cdaf \
  --eval_actor_steps 2000
```

All scripts share `codex_cdaf_common.py`, and every preset can still be modified
from the CLI. For example, `--variant codex_05_mild_cql_cdaf --cql_alpha 0.1`
works from any entrypoint.

## Research Basis

The variants borrow conservatism, behavior support, advantage-weighted
regression, in-sample value estimation, and minimalist actor extraction ideas
from these offline RL papers:

- BCQ: https://arxiv.org/abs/1812.02900
- BEAR: https://arxiv.org/abs/1906.00949
- BRAC: https://arxiv.org/abs/1911.11361
- AWR: https://arxiv.org/abs/1910.00177
- D4RL: https://arxiv.org/abs/2004.07219
- MOReL: https://arxiv.org/abs/2005.05951
- MOPO: https://arxiv.org/abs/2005.13239
- CQL: https://arxiv.org/abs/2006.04779
- AWAC: https://arxiv.org/abs/2006.09359
- CRR: https://arxiv.org/abs/2006.15134
- COMBO: https://arxiv.org/abs/2102.08363
- Fisher-BRC: https://arxiv.org/abs/2103.08050
- Decision Transformer: https://arxiv.org/abs/2106.01345
- TD3+BC: https://arxiv.org/abs/2106.06860
- EDAC: https://arxiv.org/abs/2110.01548
- IQL: https://arxiv.org/abs/2110.06169
- SPOT: https://arxiv.org/abs/2202.06239
- Diffuser: https://arxiv.org/abs/2205.09991
- MCQ: https://arxiv.org/abs/2206.04745
- Decision Diffuser: https://arxiv.org/abs/2211.15657
- XQL: https://arxiv.org/abs/2301.02328
- Cal-QL: https://arxiv.org/abs/2303.05479
- ReBRAC: https://arxiv.org/abs/2305.09836
