"""Add the energy learner to the existing ownership/pause/refill manager."""
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_onestep_deploy_v4 as previous
from utils.codex.ukjo1.error_propagation.flow_iql_research import scheduler_onestep_queue_v1 as scheduler


def allowed_modules():
    return previous.allowed_modules() | {'utils.codex.ukjo1.error_propagation.flow_iql_research.train_energy_onestep_v1'}


if __name__ == '__main__':
    scheduler.MODULES = allowed_modules()
    scheduler.main()
