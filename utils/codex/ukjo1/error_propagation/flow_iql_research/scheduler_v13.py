"""Allow one bounded numerical probe within the same sixteen GPU slots."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v12
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.codex.ukjo1.error_propagation.flow_iql_research.numerical_bc_probe_v1')

if __name__ == '__main__':
    scheduler_v4.main()
