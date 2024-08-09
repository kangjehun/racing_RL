import pathlib
import logging

import numpy as np
import tensorflow as tf

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False','True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)

def count_episodes(directory):
    filenames = directory.glob('*.npz')
    lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


def count_steps(datadir, config):
    return count_episodes(datadir)[1] * config.action_repeat

def simulate(agents, env, config, datadir, writer, prefix='train', steps=0, episodes=0, sim_state=None,
             agents_ids=None):
    if agents_ids is None:
        agents_ids = ['A']
    n_agents = len(agents_ids)
    # these are used to collect statistic of the first agent only
    cum_reward = 0.0  # episode level
    episode_progresses = []  # episode level
    max_progresses = []  # collection level
    cum_rewards = []  # collection level
    main_id = agents_ids[0]  # the agent w.r.t. we collect statistics
    # Initialize or unpack simulation state.
    if sim_state is None:
        step, episode = 0, 0
        dones = {agent_id: True for agent_id in agents_ids}
        length = np.zeros(n_agents, np.int32)
        obs = {agent_id: None for agent_id in agents_ids}
        agent_states = {agent_id: None for agent_id in agents_ids}
    else:
        step, episode, dones, length, obs, agent_states = sim_state
        cum_reward = {id: 0.0 for id in agents_ids}
    while (steps and step < steps) or (episodes and episode < episodes):
        # Reset envs if necessary.
        if any(dones.values()):
            obs = env.reset()
            if len(episode_progresses) > 0:  # at least 1 episode
                max_progresses.append(max(episode_progresses))
                cum_rewards.append(cum_reward)
            cum_reward = 0.0
        # Step agents.
        obs = {id: {k: np.stack([v]) for k, v in o.items()} for id, o in obs.items()}
        actions = dict()
        for i, agent_id in enumerate(agents_ids):
            actions[agent_id], agent_states[agent_id] = agents[i](obs[agent_id], np.stack([dones[agent_id]]),
                                                                  agent_states[agent_id])
            actions[agent_id] = np.array(actions[agent_id][0])
        assert len(actions) == len(agents_ids)
        # Step envs.
        obs, rewards, dones, infos = env.step(actions)
        # update episode-level information
        cum_reward = cum_reward + rewards[main_id]
        episode_progresses.append(infos[main_id]['lap'] + infos[main_id]['progress'] - 1)
        done = any(dones.values())
        episode += int(done)
        length += 1  # episode length until termination
        step += (int(done) * length).sum()  # num sim steps
        length *= (1 - done)
    # when the loop is over, write statistics for the 1st agent
    metrics_dict = {'progress': max_progresses,
                    'return': cum_rewards}
    summarize_collection(metrics_dict, config, datadir, writer, prefix)
    # Return new state to allow resuming the simulation.
    return (step - steps, episode - episodes, dones, length, obs, agent_states), np.mean(cum_rewards)


def summarize_collection(metrics_dict, config, datadir, writer, prefix):
    for metric_name, metric_list in metrics_dict.items():
        metrics = [(f'{prefix}/{metric_name}_mean', np.mean(metric_list)),
                   (f'{prefix}/{metric_name}_std', np.std(metric_list))]
        step = count_episodes(datadir)[1] * config.action_repeat
        with writer.as_default():  # Env might run in a different thread.
            tf.summary.experimental.set_step(step)
            [tf.summary.scalar(k, v) for k, v in metrics]
    