"""Permit the finite-proposal expectile-backup diagnostic training runner."""
from utils.flow_iql_research import scheduler_v8
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_empirical_backup_v1')

if __name__ == '__main__':
    scheduler_v4.main()
