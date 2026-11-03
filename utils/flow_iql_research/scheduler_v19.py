"""Adopt existing16learners and allow two-V-fit minibatch-reuse pilots."""
from utils.flow_iql_research import scheduler_v18
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_value_sampling_v1')

if __name__ == '__main__':
    scheduler_v4.main()
