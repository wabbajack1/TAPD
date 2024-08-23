import numpy as np
import torch

from src import utils
from src.envs import make_vec_envs
import wandb

total_num_steps_evaluation = {}
eval_episode_rewards_global = {}
episodes_global = {}

def evaluate(args, actor_critic, env_name, seed, num_processes, eval_log_dir, device, model_name, task_id=None):
    actor_critic.eval()
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True)
    
    # log init
    global total_num_steps_evaluation
    global eval_episode_rewards_global
    global episodes_global

    episodes_global.setdefault(model_name, {})
    episodes_global[model_name].setdefault(env_name, 0)

    total_num_steps_evaluation.setdefault(model_name, {})
    total_num_steps_evaluation[model_name].setdefault(env_name, 0)

    eval_episode_rewards_global.setdefault(model_name, {})
    eval_episode_rewards_global[model_name].setdefault(env_name, [])

    # log init for local variables
    eval_episode_rewards = []
    obs = eval_envs.reset()

    num_updates = int(args.eval_steps) // args.num_processes

    for steps in range(num_updates):
        with torch.no_grad():

            # check if progressive net is used or not
            if task_id is not None:
                value, action, action_log_prob, _ = actor_critic.act(obs, deterministic=True, idx=task_id)
            else:
                value, action, action_log_prob, _ = actor_critic.act(obs, deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        # log for each step
        total_num_steps_evaluation[model_name][env_name] += args.num_processes

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys(): # end of episode
                episodes_global[model_name][env_name] += 1
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_rewards_global[model_name][env_name].append(info['episode']['r']) # log for global variable
                wandb.log({f"Evaluation/Raw-Score-{env_name}-{model_name}": info['episode']['r'],
                           f"Evaluation/Timesteps-{env_name}-{model_name}": total_num_steps_evaluation[model_name][env_name],
                           f"Evaluation/Episode-{env_name}-{model_name}": episodes_global[model_name][env_name],
                           f"Evaluation/Avg-Score-{env_name}-{model_name}": np.mean(eval_episode_rewards_global[model_name][env_name][-100:])})
                
                
    eval_envs.close()

    print("Evaluation using {} episodes: mean reward {:.5f} in {} timesteps\n".format(
        episodes_global[model_name][env_name], np.mean(eval_episode_rewards_global[model_name][env_name][-100:]), total_num_steps_evaluation[model_name][env_name]))