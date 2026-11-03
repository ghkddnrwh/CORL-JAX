"""Keep16learner ownership and permit the learned mean-Q V-teacher control."""
from utils.flow_iql_research import scheduler_v20
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_mean_value_teacher_v1')

if __name__=='__main__':scheduler_v4.main()
