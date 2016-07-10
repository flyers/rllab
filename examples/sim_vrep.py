from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import rllab.misc.logger as logger
import numpy
import os
import joblib

default_log_dir = '/home/sliay/Documents/rllab/data/local/experiment/experiment_2016_07_07_693itr'
data = joblib.load(os.path.join(default_log_dir, 'params.pkl'))
policy = data['policy']
env = data['env']

rollout_num = 10
max_path_length = 500
for rollout in xrange(rollout_num):
    observations = []
    actions = []
    rewards = []
    o = env.reset()
    # env.render()
    for path_length in xrange(max_path_length):
        a, agent_info = policy.get_action(o)
        next_o, r, terminate, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        actions.append(env.action_space.flatten(a))
        rewards.append(r)
        # print 'state:', next_o
        # print 'reward:%f\n' % r
        o = next_o
        if terminate:
            break
        # env.render()

