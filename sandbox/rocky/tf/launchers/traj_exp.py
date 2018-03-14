import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahTargEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.abstractors.target_half_cheetah import COM_abstractor
from sandbox.rocky.tf.models.gen_diag_gaussian import GenDiagGaussian
from sandbox.rocky.tf.planners.lstm import LSTMplanner

env = TfEnv(normalize(HalfCheetahTargEnv))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

abstractors = [COM_abstractor]
n_layers = len(abstractors)

for i in range(n_layers):
    obs.append(abstractors[i](obs[-1]))

models = [GenDiagGaussian(name='feas_model1', abstract_dim=obs[1].shape[0], hidden_sizes=(32,))]
planners = [LSTMplanner(name='planner1', abstract_dim=obs[1].shape[0], model=models[0], hidden_dim=32)]

# algo, put in wrapper later

mbs = []
# warmup
rollouts = []
for i in range(100):
    rollout = []
    env.reset()
    for t in range(100):
        obs = env.get_current_obs()
        states = [obs]
        for i in range(n_layers):
            obs = abstractors[i](obs)
            states.append(obs)
        rollout.append(states)

        action = policy.get_action(obs)
        env.step(action)
    rollouts.append(rollout)

# fit generative model to warmup
for abs_layer in range(n_layers):
    obs = []
    nexts = []
    for rollout in rollouts:
        for t in range(len(rollout)-1):
            obs.append(rollout[t][abs_layer+1])
            nexts.append(rollout[t+1][abs_layer+1])
    models[abs_layer].fit(np.array(obs), nexts, n_steps=5000)

# fit planner
for abs_layer in range(n_layers):
    obs = []
    for rollout in rollouts:
        this_obs = []
        for t in range(len(rollout)):
            this_obs.append(rollout[t][abs_layer+1])
        obs.append(this_obs)
    planners[abs_layer].train(np.array(obs), n_steps=5000)

#repeat:
    # sample rollout
    mb = []
    for i in range(100):
        states = [[] for _ in range(n_layers+1)]
        rewards = []
        env.reset()
        obs = env.get_current_obs()
        states[0].append(obs)
        for i in range(n_layers):
            obs = abstractors[i](obs)
            states[i+1].append(obs)
        for t in range(100):
            action = policy.get_action(states[0][t])
            obs, _, _, _ = env.step(action)
            states[0].append(obs)
            # is this what we want?
            target = planners[0].forward(np.array(states[0]))
            reward = -np.linalg.norm(abstractors[0](states[0]) - target)
            rewards.append(reward)
            for i in range(n_layers):
                obs = abstractors[i](obs)
                states[i+1].append(obs)
        mb.append((rewards, states))

    # policy gradient
    # refit generative model
    # update planner
