"""Recognize the joint latent-selector runner without changing ownership logic."""
from utils.flow_iql_research import scheduler_onestep_deploy_v6 as previous
from utils.flow_iql_research import scheduler_onestep_queue_v1 as scheduler


def allowed_modules():
    return previous.allowed_modules()|{'utils.flow_iql_research.train_latent_selector_onestep_v1','utils.flow_iql_research.train_latent_order_guided_v1'}


if __name__=='__main__':
    scheduler.MODULES=allowed_modules()
    scheduler.main()
