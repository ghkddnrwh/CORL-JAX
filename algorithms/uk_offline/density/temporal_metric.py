# temporal_metric.py
#
# Learned TEMPORAL metric for DCS-IQL.
#
# Motivation (see the ceiling-test discussion): L2 observation distance picks
# the wrong neighbors on combinatorial tasks like puzzle, because task-relevant
# structure (the button configuration) is drowned out by arm pose. The oracle
# ceiling test replaces the neighbor graph with the ground-truth button_states
# graph -- but that is privileged information and not deployable.
#
# This module learns, from the dataset ALONE, a representation phi(s) in which
# temporally-reachable states are close. It uses exactly the (s, a, s', r)
# adjacency the user pointed at: every transition (s_i, s'_i) is a free label
# saying "these two states are one step apart". A contrastive (InfoNCE)
# objective pulls phi(s) toward phi(s') and pushes apart random pairs; because
# adjacency composes, states connected by short transition paths end up close
# and states that are far in the transition graph (e.g. a different button
# configuration, which takes several presses to reach) end up far -- WITHOUT
# ever being told what the button state is. This is the data-only approximation
# of the oracle graph, plugged into the same coverage_profile.metric_space hook.
#
# Design:
#   * The pure-numpy helpers (successor-index construction, k-step pair
#     sampling, embedding cache IO, standardization) are importable and unit-
#     testable WITHOUT JAX. JAX/Flax/optax are imported lazily and only the
#     encoder training needs them (mirroring the repo's optional-import style).
#   * Embeddings are L2-normalized onto the unit sphere, so Euclidean kNN in
#     coverage_profile ranks neighbors by cosine similarity (the InfoNCE
#     geometry). Do NOT per-dim standardize normalized embeddings.

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# JAX stack is optional at import time so the numpy helpers stay testable.
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax import serialization
    import optax

    _HAS_JAX = True
except Exception:  # pragma: no cover - exercised only where JAX is absent
    _HAS_JAX = False

TEMPORAL_CACHE_VERSION = 1
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Pure-numpy helpers (no JAX; unit-tested)
# ---------------------------------------------------------------------------

def build_successor_index(
    dataset: Dict[str, np.ndarray],
    boundary_tol: float = 1e-4,
) -> np.ndarray:
    """Return next_index (N,) int64 where next_index[i] = i+1 within the same
    trajectory, or -1 if transition i ends a trajectory (no in-dataset
    successor state).

    Trajectory boundaries are detected in order of preference:
      1. ``terminals``            : end where terminals[i] > 0.5
      2. ``masks``                : end where masks[i] < 0.5 (mask=0 at terminal)
      3. observation continuity   : end where next_obs[i] != obs[i+1], i.e. the
                                    stored successor does not match the next
                                    row's state (a trajectory break). This needs
                                    no flags and works on flat transition dumps.

    Only used for multi-step (horizon > 1) positive sampling; horizon == 1 uses
    next_observations directly and does not need this.
    """
    observations = np.asarray(dataset["observations"])
    n = observations.shape[0]
    end = np.zeros(n, dtype=bool)

    if "terminals" in dataset:
        end = np.asarray(dataset["terminals"]).reshape(-1) > 0.5
    elif "masks" in dataset:
        end = np.asarray(dataset["masks"]).reshape(-1) < 0.5
    elif "next_observations" in dataset and n > 1:
        nxt = np.asarray(dataset["next_observations"])
        # boundary at i (< N-1) if the stored successor of i does not match the
        # state stored at row i+1.
        mismatch = np.linalg.norm(
            nxt[:-1].astype(np.float64) - observations[1:].astype(np.float64), axis=1
        )
        end[:-1] = mismatch > boundary_tol
    end[n - 1] = True  # last row always ends its trajectory

    next_index = np.arange(1, n + 1, dtype=np.int64)
    next_index[end] = -1
    next_index[next_index >= n] = -1
    return next_index


def sample_kstep_pairs(
    next_index: np.ndarray,
    horizon: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (anchor_idx, positive_idx) index pairs for temporal contrast.

    For each anchor a random k in [1, horizon] is chosen and the successor
    pointer is followed up to k times, stopping at a trajectory end. The
    positive is the farthest index actually reached (>= 1 step whenever the
    anchor has any successor). Anchors with no successor (next_index == -1)
    yield positive == anchor (a degenerate self-positive); callers may keep
    these (harmless) or filter them.
    """
    horizon = max(int(horizon), 1)
    n = next_index.shape[0]
    anchors = rng.integers(0, n, size=batch_size)
    ks = rng.integers(1, horizon + 1, size=batch_size)

    positives = anchors.copy()
    cur = anchors.copy()
    for step in range(horizon):
        nxt = next_index[cur]
        advance = (nxt >= 0) & (step < ks)
        positives = np.where(advance, nxt, positives)
        cur = np.where(nxt >= 0, nxt, cur)
    return anchors.astype(np.int64), positives.astype(np.int64)


def standardize_embeddings(z: np.ndarray) -> np.ndarray:
    """Per-dim standardization (only for UNnormalized embeddings; normalized
    unit-sphere embeddings should be left as-is to preserve cosine geometry)."""
    z = np.asarray(z, dtype=np.float32)
    mu = z.mean(0)
    sd = z.std(0) + 1e-6
    return ((z - mu) / sd).astype(np.float32)


def temporal_signature(params: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical, hashable signature of the encoder-defining hyperparameters.
    Any change here should invalidate cached embeddings and the coverage graph
    built from them."""
    keys = (
        "cache_version", "env", "seed", "normalize", "observations_shape",
        "embed_dim", "hidden_dim", "n_hidden", "steps", "batch_size", "lr",
        "temperature", "horizon",
    )
    canonical = {}
    for key in keys:
        val = params.get(key)
        if key == "observations_shape" and val is not None:
            val = tuple(int(v) for v in val)
        canonical[key] = val
    return canonical


def signature_hash(signature: Dict[str, Any]) -> str:
    """Short deterministic hash of the CANONICAL signature (extraneous keys are
    ignored, so the filename and the stored metadata always agree)."""
    import hashlib

    canonical = temporal_signature(signature)
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def save_temporal_embeddings_cache(
    cache_path: Union[str, Path],
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata=json.dumps(temporal_signature(metadata), default=str),
    )
    print(f"Saved temporal embeddings to: {cache_path}")


def load_temporal_embeddings_cache(
    cache_path: Union[str, Path],
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Load cached embeddings; return None when missing or signature-stale."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        saved_raw = payload["metadata"]
        if hasattr(saved_raw, "item"):
            saved_raw = saved_raw.item()
        saved = temporal_signature(json.loads(str(saved_raw)))
        if expected_metadata is not None:
            expected = temporal_signature(expected_metadata)
            if saved != expected:
                print(f"Temporal cache signature mismatch, retraining: {cache_path}")
                print(f"  saved:    {saved}")
                print(f"  expected: {expected}")
                return None
        emb = np.asarray(payload["embeddings"], dtype=np.float32)
        print(f"Loaded temporal embeddings from: {cache_path}")
        return emb
    except Exception as exc:
        print(f"Failed to load temporal cache {cache_path}: {exc}; retraining.")
        return None


# ---------------------------------------------------------------------------
# Encoder + contrastive training (JAX/Flax; imported lazily)
# ---------------------------------------------------------------------------

if _HAS_JAX:

    class TemporalEncoder(nn.Module):
        """MLP encoder phi(s). Output optionally L2-normalized onto the unit
        sphere so that Euclidean kNN ranks by cosine similarity."""

        embed_dim: int
        hidden_dim: int = 256
        n_hidden: int = 2
        normalize: bool = True

        @nn.compact
        def __call__(self, s: "jnp.ndarray") -> "jnp.ndarray":
            x = s
            for _ in range(self.n_hidden):
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.relu(x)
            z = nn.Dense(self.embed_dim)(x)
            if self.normalize:
                z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + _EPS)
            return z


def _require_jax() -> None:
    if not _HAS_JAX:
        raise RuntimeError(
            "temporal_metric.train_temporal_encoder needs JAX/Flax/optax, which "
            "are not importable in this environment. Install them or run on the "
            "training machine."
        )


def train_temporal_encoder(
    observations: np.ndarray,
    next_observations: np.ndarray,
    *,
    embed_dim: int = 32,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 3e-4,
    temperature: float = 0.1,
    horizon: int = 1,
    normalize: bool = True,
    seed: int = 0,
    next_index: Optional[np.ndarray] = None,
    encode_chunk: int = 100_000,
    log_every: int = 5_000,
    device: Any = None,
) -> np.ndarray:
    """Train the contrastive temporal encoder and return phi(observations),
    an (N, embed_dim) float32 array to use as coverage_profile.metric_space.

    Positives:
      horizon == 1                -> the stored successor next_observations[i]
                                     (robust; needs no trajectory order).
      horizon  > 1 and next_index -> a within-trajectory k-step successor state
                                     obs[follow(next_index, i, k)].
    Negatives: the other elements of the same minibatch (in-batch InfoNCE).
    """
    _require_jax()
    observations = np.asarray(observations, dtype=np.float32)
    next_observations = np.asarray(next_observations, dtype=np.float32)
    n = observations.shape[0]
    use_kstep = horizon > 1 and next_index is not None
    if horizon > 1 and next_index is None:
        print(
            "temporal_metric: horizon > 1 requested but no successor index was "
            "provided; falling back to 1-step successors."
        )

    if device is None:
        device = jax.devices()[0]

    encoder = TemporalEncoder(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        normalize=normalize,
    )
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = encoder.init(init_key, jnp.zeros((1, observations.shape[1]), jnp.float32))["params"]

    tx = optax.adam(lr)
    opt_state = tx.init(params)
    inv_temp = 1.0 / max(float(temperature), _EPS)

    def _cross_entropy(logits, labels):
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(logp[jnp.arange(logits.shape[0]), labels])

    @jax.jit
    def train_step(params, opt_state, anchor_obs, positive_obs):
        def loss_fn(p):
            za = encoder.apply({"params": p}, anchor_obs)      # (B, d)
            zp = encoder.apply({"params": p}, positive_obs)    # (B, d)
            logits = (za @ zp.T) * inv_temp                    # (B, B)
            labels = jnp.arange(logits.shape[0])
            # symmetric InfoNCE: anchor->positive and positive->anchor
            loss = 0.5 * (_cross_entropy(logits, labels)
                          + _cross_entropy(logits.T, labels))
            # diagnostic: mean cosine sim of true positives
            pos_sim = jnp.mean(jnp.sum(za * zp, axis=-1))
            return loss, pos_sim

        (loss, pos_sim), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pos_sim

    rng = np.random.default_rng(seed)
    print(
        f"Training temporal encoder: N={n} dim={embed_dim} steps={steps} "
        f"batch={batch_size} horizon={horizon} temp={temperature} "
        f"positives={'k-step' if use_kstep else '1-step successor'}"
    )
    for it in range(int(steps)):
        if use_kstep:
            a_idx, p_idx = sample_kstep_pairs(next_index, horizon, batch_size, rng)
            a_obs = observations[a_idx]
            p_obs = observations[p_idx]
        else:
            a_idx = rng.integers(0, n, size=batch_size)
            a_obs = observations[a_idx]
            p_obs = next_observations[a_idx]
        params, opt_state, loss, pos_sim = train_step(
            params,
            opt_state,
            jax.device_put(jnp.asarray(a_obs), device),
            jax.device_put(jnp.asarray(p_obs), device),
        )
        if log_every and (it + 1) % log_every == 0:
            print(
                f"  temporal step {it + 1}/{steps}: infonce={float(loss):.4f} "
                f"pos_cos={float(pos_sim):.4f}"
            )

    # Encode all states in chunks.
    embeddings = np.empty((n, embed_dim), dtype=np.float32)
    apply_fn = jax.jit(lambda p, x: encoder.apply({"params": p}, x))
    for start in range(0, n, int(encode_chunk)):
        end = min(start + int(encode_chunk), n)
        emb = apply_fn(params, jax.device_put(jnp.asarray(observations[start:end]), device))
        embeddings[start:end] = np.asarray(emb, dtype=np.float32)
    return embeddings
