import numpy as np
import torch

from src import utils
from src.envs import make_vec_envs
import wandb
import logging
from src.model import Policy
from src.model import Adaptor
from src.model import BigPolicy
from src.model import IntrinsicCuriosityModule
from src.arguments import get_args


total_num_steps_evaluation = {}
eval_episode_rewards_global = {}
episodes_global = {}
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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

    
if __name__ == "__main__":

    # load the model and evaluate
    args = get_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Progress and Compress - Prediction",
        entity="agnostic",
        config=args,
        mode="online" if args.log_wandb else "disabled",
    )

    # load the model
    # set up the environment
    environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "AirRaidNoFrameskip-v4"]
    agnostic_environements = ["SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    test_environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "AirRaidNoFrameskip-v4"]

    #### init first environment for architecture initialization ####
    envs = make_vec_envs(args.env_name, args.seed, 1, args.gamma, args.log_dir, device, False)

    #### init policies ####
    print("action shape", envs.action_space)

    # load model option
    # load the model for kb learning every time, as it is required
    actor_critic_kb = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic_kb.to(device)
    
    # Load the model state from the path
    logging.info(f"Loading kb column state from the path: {args.model_path_kb}")
    checkpoint = torch.load(args.model_path_kb, map_location=device) # load the model state from the path, i.e. the index 0 is the model state
    actor_critic_kb.load_state_dict(checkpoint[0])

    evaluate(args, actor_critic_kb, "SpaceInvadersNoFrameskip-v4", args.seed, args.num_processes, args.log_dir, device, "kb")