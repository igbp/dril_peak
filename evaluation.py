import os

import numpy as np
import torch
import gym

from dril.a2c_ppo_acktr import utils
from dril.a2c_ppo_acktr.envs import make_vec_envs
DETERMINISTIC = False

def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, atari_max_steps=None, fname=None, det=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, atari_max_steps)

    if det is None:
        det = DETERMINISTIC


    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=det)

        # Obser reward and next obs
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_action = torch.clamp(action, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0]))
        else:
            clip_action = action

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(clip_action)
        #eval_envs.render()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                print(f'Episode reward: {info["episode"]["r"]}')
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    if fname == 'return_vector':
        return eval_episode_rewards
    else:

        return np.mean(eval_episode_rewards)



def evaluate2(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, atari_max_steps=None, fname=None, det=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, atari_max_steps)
    num_processes=1
    if det is None:
        det = DETERMINISTIC

    rewards = []

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=det)

        # Obser reward and next obs
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_action = torch.clamp(action, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0]))
        else:
            clip_action = action

        # Obser reward and next obs
        obs, rwd, done, infos = eval_envs.step(clip_action)
        eval_envs.render()
        rwdd = rwd.detach().clone()
        rewards.append(rwdd)


        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                print(f'Episode reward: {info["episode"]["r"]}')
                eval_episode_rewards.append(info['episode']['r'])

                file_name = f"score_hist_scr={int(info['episode']['r'])}.pt"
                dir_name = '/home/giovani/hacer/dril/dril/expert'
                file_path = os.path.join(dir_name, file_name)
                torch.save(rewards, file_path)
                print(f'Saving to {file_name}')
                rewards = []

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    if fname == 'return_vector':
        return eval_episode_rewards
    else:

        return np.mean(eval_episode_rewards)
