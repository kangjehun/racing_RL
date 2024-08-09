import argparse
import tools as tools
import random
from datetime import datetime
import pathlib

import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision

import os
import numpy as np
import time
import yaml
import logging
import sys

import wrappers as wrappers
import callbacks as callbacks

from agents.gap_follower import GapFollower

def define_config():
    """
    Default definition of command-line arguments.
    """
    config = tools.AttrDict()
    # General.
    config.datetime = datetime.now().strftime("%m-%d-%Y %H:%M:%S") # Just for logging
    config.seed = random.randint(2, 10 ** 6)
    config.logdir = pathlib.Path('logs/experiments')
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.log_videos = True
    config.gpu_growth = True
    config.precision = 32
    config.obs_type = 'lidar'
    # Environment.
    config.track = 'austria'
    config.task = 'max_progress'
    config.action_repeat = 4
    config.eval_episodes = 5
    config.time_limit_train = 2000
    config.time_limit_test = 4000
    config.prefill_agent = 'gap_follower'
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    config.clip_rewards_min = -1
    config.clip_rewards_max = 1
    # Model.
    config.encoded_obs_dim = 1080
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.reward_out_dist = 'normal'
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.pcont = True
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 1.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.3
    return config

def make_train_env(config, writer, datadir, gui=False):
    env = make_base_env(config, gui)
    if env.n_agents > 1:
        env = wrappers.FixedResetMode(env, mode='random_ball')  # sample in random points close within a ball
    else:
        env = wrappers.FixedResetMode(env, mode='random')
    env = wrappers.TimeLimit(env, config.time_limit_train / config.action_repeat)
    # storing and summary
    callback_list = []
    callback_list.append(lambda episodes: callbacks.save_episodes(datadir, episodes))
    callback_list.append(lambda episodes: callbacks.summarize_episode(episodes, config, datadir, writer, 'train'))
    env = wrappers.Collect(env, callback_list, config.precision)
    return env

def make_test_env(config, writer, datadir, gui=False):
    env = make_base_env(config, gui)
    env = wrappers.FixedResetMode(env, mode='grid')
    env = wrappers.TimeLimit(env, config.time_limit_test / config.action_repeat)
    # rendering
    render_callbacks = []
    render_callbacks.append(lambda videos: callbacks.save_videos(videos, config, datadir))
    env = wrappers.Render(env, render_callbacks)
    # summary
    callback_list = []
    callback_list.append(
        lambda episodes: callbacks.summarize_episode(episodes, config, datadir, writer, 'test'))
    env = wrappers.Collect(env, callback_list, config.precision)
    return env

def make_base_env(config, gui=False):
    env = wrappers.RaceCarBaseEnv(track=config.track, task=config.task, rendering=gui)
    env = wrappers.RaceCarWrapper(env, agent_id='A')
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
    env = wrappers.OccupancyMapObs(env)
    return env

def set_logging(config):
    # Redirect std error to file. It is useful when running experiments on cluster.
    level = logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                        filename=f"{config.logdir}/stderr.txt", filemode='a')
    stderr_logger = logging.getLogger('STDERR')
    sl = tools.StreamToLogger(stderr_logger, level)
    sys.stderr = sl

def create_log_dirs(config):
    # create filename
    prefix = f"{config.track}_dreamer_{config.task}"
    model_archs = f"{config.obs_type.replace('_', '')}_{config.action_dist.replace('_', '')}"
    params = f"AR{config.action_repeat}_BL{config.batch_length}_H{config.horizon}"
    suffix = f"{config.seed}_{time.time()}"
    # create log dirs
    logdir = pathlib.Path(f'{config.logdir}/{prefix}_{model_archs}_{params}_{suffix}')
    datadir = logdir / 'episodes'               # where storing the episodes as np files
    checkpoint_dir = logdir / 'checkpoints'     # where storing the model checkpoints
    best_checkpoint_dir = logdir / 'best'       # where storing the best model checkpoints
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # save config
    with open(logdir / 'config.yaml', 'w') as file:
        yaml.dump(vars(config), file)
    return logdir, datadir, checkpoint_dir

def setup_experiments(config):
    # GPU
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))
    # seeding
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

def main(config):
    # Setup logging
    setup_experiments(config)
    config.steps = int(config.steps)
    config.logdir, datadir, checkpoint_dir = create_log_dirs(config)
    # set_logging(config)
    writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    print(f"[Info] Logdir {config.logdir}")
    # Create environments
    train_env = make_train_env(config, writer, datadir, gui=False)
    test_env  = make_test_env(config, writer, datadir, gui=False)
    agent_ids = train_env.agent_ids
    actspace = train_env.action_space
    obspace = train_env.observation_space

    # Prefill phase
    step = tools.count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'[Info] Prefill dataset (strategy={config.prefill_agent}) with {prefill} steps.')
    # Choose prefill strategy.
    if config.prefill_agent == 'random':
        # Prefill strategy: random actions
        random_agent = lambda o, d, s: ([train_env.action_space[agent_ids[0]].sample()], None)
        agents = [random_agent for _ in range(train_env.n_agents)]
    elif config.prefill_agent == 'gap_follower':
        # Prefil strategy: FTG with a fixed low speed (negative value because of shifting in 0,1)
        ftg = GapFollower()
        fix_speed = -0.96
        gap_follower_agent = lambda o, d, s: ([np.clip(np.array([fix_speed, ftg.action(o)[-1]]), -1, +1)], None)
        agents = [gap_follower_agent for _ in range(train_env.n_agents)]
    else:
        raise NotImplementedError(f'prefill agent {config.prefill_agent} not implemented')
    # Run prefill simulations
    tools.simulate(agents, train_env, config, datadir, writer, prefix='prefill',
                   steps=prefill / config.action_repeat, agents_ids = agent_ids)
    writer.flush()

    # Initialize Dreamer model

    # Train and Evaluate the agent over the simulation process

if __name__ == '__main__':
    print("Initializing dream.py...")
    parser = argparse.ArgumentParser(description='Jehun Dreamer')
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    args = parser.parse_args()
    main(args)