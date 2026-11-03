"""Allow the AWR mixture policy pilot within the unchanged sixteen GPU slots."""
from utils.flow_iql_research import scheduler_v14
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_awr_mixture_v1')

if __name__=='__main__':scheduler_v4.main()
