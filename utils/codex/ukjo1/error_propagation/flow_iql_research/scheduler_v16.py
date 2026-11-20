"""Add a uniform half empirical-expectile correction to the existing 16 slots."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v15
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.codex.ukjo1.error_propagation.flow_iql_research.train_half_empirical_backup_v1')

if __name__ == '__main__':
    scheduler_v4.main()
