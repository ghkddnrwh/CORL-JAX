"""Allow the real-data rank-weighted flow pilot in the same sixteen slots."""
from utils.flow_iql_research import scheduler_v13
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_data_rank_flow_v1')

if __name__ == '__main__':
    scheduler_v4.main()
