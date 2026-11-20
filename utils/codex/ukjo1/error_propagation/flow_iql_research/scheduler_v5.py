"""Extend the ownership-checked scheduler to the independent-V control."""
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_independent_value')

if __name__ == '__main__':
    scheduler_v4.main()
