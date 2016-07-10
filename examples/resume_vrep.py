from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import rllab.misc.logger as logger
from rllab import config
import numpy
import os
import datetime
import dateutil.tz
import ast
import uuid
import base64
import joblib

default_log_dir = '/home/sliay/Documents/rllab/data/local/experiment'
now = datetime.datetime.now(dateutil.tz.tzlocal())
# avoid name clashes when running distributed jobs
rand_id = str(uuid.uuid4())[:5]
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
log_dir = os.path.join(default_log_dir, default_exp_name)
tabular_log_file = os.path.join(log_dir, 'progress.csv')
text_log_file = os.path.join(log_dir, 'debug.log')
params_log_file = os.path.join(log_dir, 'params.json')

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode('last')
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % default_exp_name)

last_snapshot_dir = '/home/sliay/Documents/rllab/data/local/experiment/experiment_2016_07_07_498itr'
data = joblib.load(os.path.join(last_snapshot_dir, 'params.pkl'))
policy = data['policy']
env = data['env']
baseline = data['baseline']
# env = normalize(GymEnv("VREP-v0", record_video=False))

# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     The neural network policy should have two hidden layers, each with 32 hidden units.
#     hidden_sizes=(128, 128)
# )
# print('policy initialization')
# print(policy.get_param_values())
# policy.set_param_values(numpy.load(os.path.join(last_snapshot_dir, 'policy.npy')))
# print('----------------------')
# print(policy.get_param_values())

# baseline = LinearFeatureBaseline(env_spec=env.spec)
# baseline.set_param_values(numpy.load(os.path.join(last_snapshot_dir, 'baseline.npy')))

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    # batch_size=4000,
    batch_size=1000,
    max_path_length=env.horizon,
    n_itr=500,
    discount=0.99,
    step_size=0.0025,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)
algo.train()

logger.remove_tabular_output(tabular_log_file)
logger.remove_text_output(text_log_file)
logger.pop_prefix()
# run_experiment_lite(
#     algo.train(),
#     # Number of parallel workers for sampling
#     n_parallel=1,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     seed=1,
#     # plot=True,
# )
