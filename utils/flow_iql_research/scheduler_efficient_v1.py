"""Dedicated16-slot efficiency queue with frozen ownership/adoption checks."""
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.add('utils.flow_iql_research.train_efficient_v1')

if __name__=='__main__':scheduler_v4.main()
