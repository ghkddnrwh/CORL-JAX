"""Allow runners with fewer auxiliary evaluations; primary protocol is fixed."""
from utils.flow_iql_research import scheduler_v5
from utils.flow_iql_research import scheduler_v4

scheduler_v4.MODULES.update({
    'utils.flow_iql_research.train_policy_v2',
    'utils.flow_iql_research.train_independent_value_v2',
})

if __name__ == '__main__':
    scheduler_v4.main()
