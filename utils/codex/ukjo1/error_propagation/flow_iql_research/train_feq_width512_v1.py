"""Use the existing checkpointed FEQ training loop with explicit width 512."""
import fcntl
from pathlib import Path
import sys
from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_width512_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research import train_discount_0995_n9_v1 as runner


def main():
    # A lock protects replay RNG/checkpoints if a scheduler is restarted.
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.core = core
        runner.return_bounds = core.return_bounds
        runner.main()


if __name__ == '__main__':
    main()
