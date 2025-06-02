from .dqn_basic_config import make_dqn_basic_config, make_logger
from .ppo_basic_config import make_ppo_basic_config

config_map = {
    "dqn_basic": make_dqn_basic_config,
    "ppo_basic": make_ppo_basic_config
}