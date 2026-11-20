"""Two implementation allowlist; underlying ownership/2x8 logic unchanged."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_v4


def allowed_modules():
    return scheduler_v4.MODULES|{'utils.codex.ukjo1.error_propagation.flow_iql_research.train_onestep_deploy_v1',
                               'utils.codex.ukjo1.error_propagation.flow_iql_research.train_set_onestep_deploy_v1'}


if __name__=='__main__':
    scheduler_v4.MODULES=allowed_modules()
    scheduler_v4.main()
