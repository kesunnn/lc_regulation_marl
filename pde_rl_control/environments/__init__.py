from gym.envs.registration import register

register(id='TrafficEnv_lane5_1', entry_point='pde_rl_control.environments.lane5_1.traffic_env:Traffic_Env')
register(id='TrafficEnv_lane5_2', entry_point='pde_rl_control.environments.lane5_2.traffic_env:Traffic_Env_Four_Action')
register(id='TrafficEnv_lane5_3', entry_point='pde_rl_control.environments.lane5_3.traffic_env:Traffic_Env_Four_Action_With_Baseline')