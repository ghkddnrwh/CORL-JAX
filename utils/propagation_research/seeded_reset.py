"""Control ManipSpace's fresh-Box reset RNG in frozen evaluations only.

Some installed OGBench environments construct a new action space on every
property access. Seeding one temporary Box does not seed the Boxes sampled by
their reset warmup. Cache an equivalent Box on this environment instance only;
installed sources, live training, action bounds, and reward functions are intact.
"""

import hashlib
import inspect
from pathlib import Path


PROTOCOL = "frozen_cpu50_cached_action_space_seeded_v1"


def prepare_seeded_action_space(env, seed):
    import numpy as np

    raw = env.unwrapped
    original = type(raw)
    first, second = raw.action_space, raw.action_space
    fresh = first is not second
    # Use the outer space as well, in case an existing wrapper caches its space.
    space = env.action_space
    for other in (first, second):
        if (space.shape != other.shape or space.dtype != other.dtype
                or not np.array_equal(space.low, other.low)
                or not np.array_equal(space.high, other.high)):
            raise ValueError("Cannot cache inconsistent action-space bounds")
    if fresh:
        if hasattr(raw, "_propagation_eval_action_space"):
            raise ValueError("Frozen-evaluation action space is already prepared")
        raw._propagation_eval_action_space = space

        def get_space(self):
            return self._propagation_eval_action_space

        raw.__class__ = type("SeededEvaluation" + original.__name__, (original,),
                             {"action_space": property(get_space)})
    space.seed(int(seed))
    if raw.action_space is not raw.action_space or env.action_space is not space:
        raise ValueError("Action-space seeding did not persist through wrappers")
    source = Path(inspect.getfile(original))
    digest = hashlib.sha256()
    for x in (space.low, space.high):
        digest.update(np.asarray(x).tobytes())
    return {"seed": int(seed), "native_action_space_property_was_fresh": fresh,
            "cached_only_on_this_env_instance": fresh,
            "original_environment_class": original.__module__ + "." + original.__name__,
            "environment_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "action_space_bounds_sha256": digest.hexdigest(),
            "action_shape": list(space.shape), "action_dtype": str(space.dtype),
            "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
