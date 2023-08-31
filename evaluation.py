import numpy as np
import torch

from new_a2c import utils
from new_a2c.envs import make_vec_envs
import wandb

total_num_steps_evaluation = {}

def evaluate(args, actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir, device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)
    
    global total_num_steps_evaluation

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
    eval_episode_rewards = []
    obs = eval_envs.reset()

    num_updates = int(args.eval_steps) // args.num_processes

    for steps in range(num_updates):
        with torch.no_grad():
            value, action, action_log_prob = actor_critic.act(obs,deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        total_num_steps_evaluation.setdefault(env_name, 0)
        total_num_steps_evaluation[env_name] += args.num_processes

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                wandb.log({f"Evaluation/Score-{env_name}": info['episode']['r'],
                           f"Evaluation/Timesteps-{env_name}": total_num_steps_evaluation[env_name]})

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))