"""Enable observed-next-action anchored backups, preserving active workers."""
from utils.flow_iql_research import scheduler_v9
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_anchored_empirical_backup_v1')

if __name__ == '__main__':
    scheduler_v4.main()
