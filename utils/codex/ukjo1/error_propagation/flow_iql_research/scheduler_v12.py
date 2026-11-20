"""Enable the policy flow trained on ranked behavior-action targets."""
from utils.flow_iql_research import scheduler_v11
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_rank_flow_v2')

if __name__ == '__main__':
    scheduler_v4.main()
