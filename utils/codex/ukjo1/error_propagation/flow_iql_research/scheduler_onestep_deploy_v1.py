"""Versioned 2x8 scheduler whitelist; preserve all historical schedulers."""
from utils.flow_iql_research import scheduler_v4


if __name__=='__main__':
    scheduler_v4.MODULES=scheduler_v4.MODULES|{'utils.flow_iql_research.train_onestep_deploy_v1'}
    scheduler_v4.main()
