"""Enable the mean-target finite-expectile backup control."""
from utils.flow_iql_research import scheduler_v10
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_mean_empirical_backup_v1')

if __name__ == '__main__':
    scheduler_v4.main()
