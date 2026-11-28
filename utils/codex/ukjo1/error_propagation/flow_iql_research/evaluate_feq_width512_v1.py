"""Preserve the existing CPU50 evaluation protocol for width-512 FEQ."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_width512_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research import evaluate_discount_0995_n9_v1 as runner


if __name__ == '__main__':
    runner.core = core
    runner.main()
