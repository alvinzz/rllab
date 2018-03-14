from rllab.envs.mujoco.half_cheetah_env import HalfCheetahTargEnv

def COM_abstractor(hc_targ_state):
    return hc_targ_state[:-3]
