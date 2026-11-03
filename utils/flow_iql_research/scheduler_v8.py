"""Allow the bounded expectile-weighted flow runner in the persistent queue."""
from utils.flow_iql_research import scheduler_v7
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_expectile_flow_v1')

if __name__ == '__main__':
    scheduler_v4.main()
