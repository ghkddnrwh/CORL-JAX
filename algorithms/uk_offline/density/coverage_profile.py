# coverage_profile.py
#
# One-time, nonparametric "local coverage profile" for DCS-IQL
# (Density-Certified Stitching IQL).
#
# For every transition i in an offline dataset this module precomputes:
#   - neighbor_indices[i]  : indices of the k nearest states (self excluded)
#   - neighbor_distances[i]: distances to those neighbors
#   - knn_radius[i]        : distance to the k-th neighbor  -> density rho(s)
#   - action_spread[i]     : RMS spread of neighbor actions -> b_act(s)
#   - disp_dispersion[i]   : directional dispersion of neighbor next-state
#                            displacements                  -> b_dir(s)
#
# Post-processing helpers turn these raw statistics into:
#   - density / diversity confidences in [0, 1] (percentile scaling)
#   - junction score J(s) = density x diversity
#   - gate(s) in [0, 1] (ramp between two percentiles of J)
#   - per-neighbor kernel weights exp(-(d/h)^2) with per-state bandwidth h
#
# Design notes:
#   * Raw statistics are cached; percentile scaling, junction mode, gate
#     percentiles, and kernel bandwidth are cheap post-processing applied at
#     load time, so changing those hyperparameters does NOT invalidate caches.
#   * numpy/scipy only: importable without JAX (used by both the trainer and
#     the standalone Phase-0 analysis script).

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

COVERAGE_CACHE_VERSION = 3
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_coverage_profile(
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    k: int = 8,
    subsample_size: int = 10_000_000,
    chunk_size: int = 50_000,
    seed: int = 0,
    dynamics_ridge: float = 1e-3,
    metric_space: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute the local coverage profile of an offline dataset.

    Args:
        observations:      (N, Ds) states, ideally already normalized with the
                           same statistics used at training time.
        actions:           (N, Da) dataset actions.
        next_observations: (N, Ds) successor states (same normalization).
        k:                 number of neighbors kept per state (self excluded).
        subsample_size:    cap on the number of reference points used to build
                           the KD-tree. Neighbor indices always refer to the
                           FULL dataset (mapped through the subsample).
        chunk_size:        query/gather chunk size (memory control).
        seed:              rng seed for the reference subsample.
        dynamics_ridge:    ridge regularization for the per-state local linear
                           dynamics fit used to score neighbor dynamics
                           consistency (numerical stability only; the residual
                           is later compared to its own per-row median, so the
                           absolute ridge scale cancels).
        metric_space:      optional (N, Dm) array used ONLY to select neighbors
                           and measure neighbor_distances. When None (default)
                           neighbors are selected in observation space, the
                           current behavior. Supplying an oracle / learned
                           representation here selects neighbors in THAT space
                           while the physical statistics (action_spread,
                           disp_dispersion, dynamics_residual) are still computed
                           from the real actions/observations of the selected
                           neighbors. This is the single hook used by the oracle
                           ceiling test and by any future learned-metric variant.

    Returns dict with keys:
        neighbor_indices  (N, k) int32   global dataset indices, self excluded
        neighbor_distances(N, k) float32 distances in the metric space
        knn_radius        (N,)   float32 distance to the k-th kept neighbor
        action_spread     (N,)   float32 RMS radius of neighbor actions
        disp_dispersion   (N,)   float32 1 - resultant/total of neighbor unit
                                          next-state displacements, in [0, 1]
        dynamics_residual (N, k) float32 ||eps_j|| of neighbor j under the
                                          local linear action->displacement fit
                                          over {i} u N_k(s_i); small => neighbor
                                          j shares s_i's local dynamics regime
    """
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    next_observations = np.asarray(next_observations, dtype=np.float32)

    n = observations.shape[0]
    if n == 0:
        return {
            "neighbor_indices": np.zeros((0, k), dtype=np.int32),
            "neighbor_distances": np.zeros((0, k), dtype=np.float32),
            "knn_radius": np.zeros((0,), dtype=np.float32),
            "action_spread": np.zeros((0,), dtype=np.float32),
            "disp_dispersion": np.zeros((0,), dtype=np.float32),
            "dynamics_residual": np.zeros((0, k), dtype=np.float32),
        }
    if actions.shape[0] != n or next_observations.shape[0] != n:
        raise ValueError("observations/actions/next_observations must share N")

    # The metric space drives neighbor selection; physical stats use the real
    # observations/actions/next_observations regardless.
    if metric_space is None:
        metric = observations
    else:
        metric = np.asarray(metric_space, dtype=np.float32)
        if metric.shape[0] != n:
            raise ValueError("metric_space must have the same N as observations")
        if metric.ndim != 2:
            raise ValueError("metric_space must be 2-D (N, Dm)")

    from scipy.spatial import cKDTree  # imported lazily; hard requirement here

    rng = np.random.default_rng(seed)
    ref_size = int(min(n, subsample_size))
    if ref_size < n:
        ref_idx = rng.choice(n, size=ref_size, replace=False)
    else:
        ref_idx = np.arange(n)
    ref_idx = np.asarray(ref_idx, dtype=np.int64)

    # We can keep at most ref_size - 1 neighbors after self-exclusion.
    k_eff = int(min(k, max(ref_size - 1, 1)))
    if k_eff < k:
        print(
            f"coverage_profile: reference set too small for k={k}; "
            f"clamping to k={k_eff}."
        )

    tree = cKDTree(metric[ref_idx])
    query_k = int(min(k_eff + 1, ref_size))

    neighbor_indices = np.empty((n, k_eff), dtype=np.int32)
    neighbor_distances = np.empty((n, k_eff), dtype=np.float32)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        dists, local_idx = tree.query(metric[start:end], k=query_k, workers=-1)
        if query_k == 1:
            dists = dists[:, None]
            local_idx = local_idx[:, None]
        global_idx = ref_idx[local_idx]  # (c, query_k) indices into full dataset

        # Self-exclusion: mask out the entry equal to the query's own index
        # (present iff the query point is in the reference set), then keep the
        # k_eff nearest remaining entries. Duplicated states at distance 0 that
        # are NOT the query itself are legitimately kept as neighbors.
        query_global = np.arange(start, end, dtype=np.int64)[:, None]
        self_mask = global_idx == query_global
        dists_masked = np.where(self_mask, np.inf, dists)
        order = np.argsort(dists_masked, axis=1, kind="stable")[:, :k_eff]
        neighbor_indices[start:end] = np.take_along_axis(
            global_idx, order, axis=1
        ).astype(np.int32)
        neighbor_distances[start:end] = np.take_along_axis(
            dists_masked, order, axis=1
        ).astype(np.float32)

    if not np.all(np.isfinite(neighbor_distances)):
        # Can only happen in pathological tiny-reference cases.
        bad = ~np.isfinite(neighbor_distances)
        max_finite = np.nanmax(np.where(bad, np.nan, neighbor_distances))
        max_finite = float(max_finite) if np.isfinite(max_finite) else 1.0
        neighbor_distances[bad] = max_finite
        print("coverage_profile: replaced non-finite neighbor distances.")

    knn_radius = neighbor_distances[:, -1].copy()

    # Unit next-state displacements for directional dispersion. Zero-length
    # displacements (static transitions) contribute nothing to either the
    # resultant or the total length.
    disp = next_observations - observations
    disp_norm = np.linalg.norm(disp, axis=-1)
    unit_disp = disp / np.maximum(disp_norm[:, None], _EPS)
    unit_disp = np.where(disp_norm[:, None] > _EPS, unit_disp, 0.0).astype(np.float32)
    moving = (disp_norm > _EPS).astype(np.float32)

    action_dim = actions.shape[1]
    # The local linear dynamics fit estimates a (Da -> Ds) map from the pool of
    # {self} u neighbors. With pool size 1 + k_eff and Da free directions, the
    # residuals are only informative once 1 + k_eff comfortably exceeds Da.
    if (1 + k_eff) <= action_dim + 1:
        print(
            f"coverage_profile: pool size {1 + k_eff} is small relative to action "
            f"dim {action_dim}; dynamics residuals will be near-zero and the "
            f"dynamics gate uninformative. Consider increasing k to >= ~2*action_dim."
        )

    ridge = float(dynamics_ridge)
    action_spread = np.empty((n,), dtype=np.float32)
    disp_dispersion = np.empty((n,), dtype=np.float32)
    dynamics_residual = np.empty((n, k_eff), dtype=np.float32)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        idx = neighbor_indices[start:end]  # (c, k_eff)

        nbr_actions = actions[idx]  # (c, k_eff, Da)
        centroid = nbr_actions.mean(axis=1, keepdims=True)
        sq = ((nbr_actions - centroid) ** 2).sum(axis=-1)  # (c, k_eff)
        action_spread[start:end] = np.sqrt(sq.mean(axis=1))

        nbr_unit = unit_disp[idx]  # (c, k_eff, Ds)
        resultant = np.linalg.norm(nbr_unit.sum(axis=1), axis=-1)  # (c,)
        total = moving[idx].sum(axis=1)  # (c,) number of moving neighbors
        dispersion = 1.0 - resultant / np.maximum(total, _EPS)
        # If (almost) no neighbor moves, there is no movement variety at all.
        dispersion = np.where(total > 0.5, dispersion, 0.0)
        disp_dispersion[start:end] = np.clip(dispersion, 0.0, 1.0)

        # ----- per-edge dynamics consistency g_ij ---------------------------
        # Fit a per-state local linear map  Delta_s ~ A (a - a_bar) + b_bar
        # over the pool {self} u neighbors, then score each neighbor by how
        # well its own transition is explained by that map. A neighbor from a
        # different dynamics regime (similar action, different displacement)
        # gets a large residual and will be down-weighted in the pool.
        #
        # The residual is normalized by the local displacement scale so that it
        # is dimensionless and ABSOLUTE: a residual that is a small fraction of
        # the typical neighborhood displacement means the neighbor shares the
        # local dynamics; a residual comparable to the displacement itself means
        # it does not. (A per-row-relative scale would wash out exactly the
        # cross-region contrast we need, e.g. coherent corridor vs incoherent
        # puzzle neighborhood.)
        self_a = actions[start:end][:, None, :]              # (c, 1, Da)
        self_d = disp[start:end][:, None, :]                 # (c, 1, Ds)
        pool_a = np.concatenate([self_a, nbr_actions], axis=1)        # (c, P, Da)
        pool_d = np.concatenate([self_d, disp[idx]], axis=1)         # (c, P, Ds)
        a_mean = pool_a.mean(axis=1, keepdims=True)
        d_mean = pool_d.mean(axis=1, keepdims=True)
        x = pool_a - a_mean                                  # (c, P, Da)
        y = pool_d - d_mean                                  # (c, P, Ds)
        xtx = np.einsum("cpa,cpb->cab", x, x)                # (c, Da, Da)
        xty = np.einsum("cpa,cpd->cad", x, y)                # (c, Da, Ds)
        xtx = xtx + ridge * np.eye(action_dim, dtype=xtx.dtype)[None]
        weight_map = np.linalg.solve(xtx, xty)               # (c, Da, Ds)
        y_hat = np.einsum("cpa,cad->cpd", x, weight_map)     # (c, P, Ds)
        residual = np.linalg.norm(y - y_hat, axis=-1)        # (c, P)

        pool_disp_norm = np.linalg.norm(pool_d, axis=-1)     # (c, P)
        disp_scale = np.median(pool_disp_norm, axis=1, keepdims=True)  # (c, 1)
        moving_region = disp_scale > _EPS
        normalized = np.where(
            moving_region, residual / np.maximum(disp_scale, _EPS), 0.0
        )
        dynamics_residual[start:end] = normalized[:, 1:].astype(np.float32)

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
        "knn_radius": knn_radius.astype(np.float32),
        "action_spread": action_spread,
        "disp_dispersion": disp_dispersion,
        "dynamics_residual": dynamics_residual,
    }


# ---------------------------------------------------------------------------
# Post-processing (cheap; applied at load time, never cached)
# ---------------------------------------------------------------------------

def percentile_confidence(
    values: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    invert: bool = False,
) -> np.ndarray:
    """Map raw statistics to [0, 1] via percentile min-max scaling.

    invert=True flips the direction (used for knn_radius: small radius means
    dense, hence high confidence).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.reshape(-1)
    lo = np.percentile(values, percentile_low)
    hi = np.percentile(values, percentile_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        conf = np.ones_like(values, dtype=np.float32)
    else:
        conf = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return (1.0 - conf) if invert else conf


def build_junction_score(
    density_conf: np.ndarray,
    action_div_conf: np.ndarray,
    disp_div_conf: np.ndarray,
    mode: str = "product",
) -> np.ndarray:
    """J(s) = density x diversity. `mode` selects the diversity signal:
    'action', 'displacement', or 'product' (geometric mean of both)."""
    if mode == "action":
        diversity = action_div_conf
    elif mode == "displacement":
        diversity = disp_div_conf
    elif mode == "product":
        diversity = np.sqrt(
            np.clip(action_div_conf, 0.0, 1.0) * np.clip(disp_div_conf, 0.0, 1.0)
        )
    else:
        raise ValueError(f"Unknown diversity mode: {mode}")
    return (np.clip(density_conf, 0.0, 1.0) * diversity).astype(np.float32)


def gate_from_junction(
    junction: np.ndarray,
    gate_low_percentile: float = 60.0,
    gate_high_percentile: float = 95.0,
) -> np.ndarray:
    """Ramp gate(s) in [0, 1]: 0 below the low percentile of J, 1 above the
    high percentile, linear in between."""
    junction = np.asarray(junction, dtype=np.float32)
    if junction.size == 0:
        return junction
    lo = float(np.percentile(junction, gate_low_percentile))
    hi = float(np.percentile(junction, gate_high_percentile))
    if hi <= lo:
        # Degenerate score distribution: the old behavior returned all ones
        # when junction == 0 everywhere, which accidentally enabled every
        # neighbor edge in exactly the regime where the certificate carries no
        # information. Treat a near-zero degenerate score as "no certified
        # support" and return all zeros. If all scores are tied at a positive
        # value, return all ones because every point is equally certified.
        if hi <= _EPS:
            return np.zeros_like(junction, dtype=np.float32)
        return np.ones_like(junction, dtype=np.float32)
    return np.clip((junction - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def neighbor_kernel_weights(
    neighbor_distances: np.ndarray,
    bandwidth_scale: float = 1.0,
    bandwidth_floor_frac: float = 1.0,
) -> np.ndarray:
    """Per-neighbor distance kernel exp(-(d/h)^2) with self-tuning bandwidth.

    Per-state bandwidth is the local neighbor-distance median, floored at a
    global scale:

        h_i = bandwidth_scale * max( median_j d_ij ,
                                     bandwidth_floor_frac * global_scale )

    where ``global_scale`` is the median of all POSITIVE neighbor distances in
    the dataset (exact-duplicate 0-distance edges are excluded so the global
    scale reflects the real resolvable distance of the metric space).

    Why the floor: without it, ``h_i`` collapses toward 0 whenever a state's
    neighbors are (near-)duplicates. That happens systematically when the
    metric space is DISCRETE -- e.g. the oracle button_states graph, where a
    state's nearest button-neighbors sit at distance ~0 -- so the local median
    is 0, ``h_i`` hits the numerical ``_EPS`` floor, and the kernel degenerates
    to all-or-nothing (weight 1 at d=0, weight 0 at any d>0). The floor keeps
    ``h_i`` at a meaningful global scale, so mixed neighborhoods (some exact
    matches, some near matches) get a graded kernel instead of a hard 0/1 cut.

    IMPORTANT / honest scope: the floor cannot help a row whose neighbors are
    ALL at distance exactly 0 (all exact button matches). There exp(-(0/h)^2)=1
    for every h>0, so that row still hands uniform weight 1 to every neighbor,
    no matter the floor -- distance-0 simply carries no information to grade on.
    Controlling how much such a saturated pool influences training is the job of
    the neighbor-mass weights (eta_V / eta_pi), NOT of this kernel. The floor is
    correct, necessary hygiene for continuous and mixed metric spaces; it is not
    by itself sufficient to tame the fully-degenerate discrete case.

    bandwidth_floor_frac=0.0 recovers the previous (unfloored) behavior.
    """
    d = np.asarray(neighbor_distances, dtype=np.float32)
    if d.size == 0:
        return d
    local = np.median(d, axis=1, keepdims=True)

    floor = 0.0
    if bandwidth_floor_frac > 0.0:
        positive = d[d > _EPS]
        if positive.size > 0:
            floor = float(bandwidth_floor_frac) * float(np.median(positive))

    h = bandwidth_scale * np.maximum(local, floor)
    h = np.maximum(h, _EPS).astype(np.float32)
    return np.exp(-((d / h) ** 2)).astype(np.float32)


def dynamics_consistency_weights(
    dynamics_residual: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Per-edge dynamics-consistency gate g_ij in [0, 1].

    Operates on the displacement-normalized residual produced by
    compute_coverage_profile (residual ||eps_j|| divided by the local
    displacement scale), so the residual is already dimensionless and
    comparable across states:

        g_ij = exp(-(r_ij / scale)^2)

    A neighbor whose transition is well explained by s_i's local
    action->displacement map has r ~ 0 => g ~ 1; a neighbor from a different
    dynamics regime has r of order 1 (residual comparable to the displacement
    itself) => g small. `scale` is the gate softness: r = scale maps to
    g = exp(-1) ~ 0.37, so larger `scale` is a gentler gate. r == 0 rows
    (perfectly coherent, or static neighborhoods that were zeroed at compute
    time) yield g = 1, i.e. the gate abstains.
    """
    r = np.asarray(dynamics_residual, dtype=np.float32)
    if r.size == 0:
        return r
    scale = max(float(scale), _EPS)
    return np.exp(-((r / scale) ** 2)).astype(np.float32)


def summarize_profile(
    density_conf: np.ndarray,
    action_div_conf: np.ndarray,
    disp_div_conf: np.ndarray,
    junction: np.ndarray,
    gate: np.ndarray,
    dynamics_g: Optional[np.ndarray] = None,
) -> str:
    def _s(x):
        return f"mean={float(np.mean(x)):.3f} std={float(np.std(x)):.3f}"

    dense_homog = float(np.mean((density_conf > 0.7) & (action_div_conf < 0.3)))
    parts = [
        f"density [{_s(density_conf)}]",
        f"action_div [{_s(action_div_conf)}]",
        f"disp_div [{_s(disp_div_conf)}]",
        f"junction [{_s(junction)}]",
        f"gate [{_s(gate)} active={float(np.mean(gate > 0)):.3f}]",
    ]
    if dynamics_g is not None:
        parts.append(f"dyn_g [{_s(dynamics_g)}]")
    parts.append(f"dense-but-homogeneous frac={dense_homog:.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Cache IO (raw statistics only; metadata-validated like the DC-IQL cache)
# ---------------------------------------------------------------------------

_PROFILE_ARRAY_KEYS = (
    "neighbor_indices",
    "neighbor_distances",
    "knn_radius",
    "action_spread",
    "disp_dispersion",
    "dynamics_residual",
)


def canonicalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    canonical = dict(metadata)
    for key in ("observations_shape", "actions_shape"):
        if key in canonical and canonical[key] is not None:
            canonical[key] = tuple(int(v) for v in canonical[key])
    return canonical


def save_coverage_profile_cache(
    cache_path: Union[str, Path],
    profile: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(profile[key]) for key in _PROFILE_ARRAY_KEYS}
    np.savez_compressed(
        cache_path,
        metadata=json.dumps(canonicalize_metadata(metadata)),
        **payload,
    )
    print(f"Saved coverage profile to: {cache_path}")


def load_coverage_profile_cache(
    cache_path: Union[str, Path],
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load a cached profile. Returns None when missing or metadata-stale.
    expected_metadata=None skips validation (analysis-script use)."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        saved_meta_raw = payload["metadata"]
        if hasattr(saved_meta_raw, "item"):
            saved_meta_raw = saved_meta_raw.item()
        saved_metadata = canonicalize_metadata(json.loads(str(saved_meta_raw)))

        if expected_metadata is not None:
            expected = canonicalize_metadata(expected_metadata)
            if saved_metadata != expected:
                print(f"Coverage cache metadata mismatch, recomputing: {cache_path}")
                print(f"  saved:    {saved_metadata}")
                print(f"  expected: {expected}")
                return None

        profile = {key: np.asarray(payload[key]) for key in _PROFILE_ARRAY_KEYS}
        n = profile["knn_radius"].shape[0]
        k = profile["neighbor_indices"].shape[1] if n > 0 else 0
        if profile["neighbor_indices"].shape != (n, k) or (
            profile["neighbor_distances"].shape != (n, k)
        ):
            print("Coverage cache shape mismatch, recomputing.")
            return None
        profile["__metadata__"] = saved_metadata  # type: ignore[assignment]
        print(f"Loaded coverage profile from: {cache_path}")
        return profile
    except Exception as exc:
        print(f"Failed to load coverage cache {cache_path}: {exc}; recomputing.")
        return None
