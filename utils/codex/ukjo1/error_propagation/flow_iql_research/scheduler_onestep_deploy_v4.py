"""Two implementation allowlist with owned benchmark-drain pause support."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_onestep_queue_v1 as scheduler_v4


def allowed_modules():
    return scheduler_v4.MODULES|{'utils.codex.ukjo1.error_propagation.flow_iql_research.train_onestep_deploy_v1',
                               'utils.codex.ukjo1.error_propagation.flow_iql_research.train_set_onestep_deploy_v1',
                               'utils.codex.ukjo1.error_propagation.flow_iql_research.train_mean_teacher_onestep_v1'}


if __name__=='__main__':
    scheduler_v4.MODULES=allowed_modules()
    scheduler_v4.main()
