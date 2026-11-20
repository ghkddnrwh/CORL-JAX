"""Extend the persistent queue to the value-hinge student policy control."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v6
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.codex.ukjo1.error_propagation.flow_iql_research.train_aligned_policy')

if __name__ == '__main__':
    scheduler_v4.main()
