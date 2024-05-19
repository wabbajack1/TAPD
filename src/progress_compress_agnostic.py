import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import algo, utils
from src.arguments import get_args
from src.envs import make_vec_envs
from src.model import Policy, BigPolicy, Adaptor, IntrinsicCuriosityModule
from src.storage import RolloutStorage
from evaluation import evaluate
import wandb
from src.utils import freeze_everything, unfreeze_everything
from datetime import datetime

# total timesteps across all visits
total_num_steps_progess = {}
total_num_steps_compress = {}
total_num_steps_agnostic = {}


def progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name, vis):
    """The progress phase, where the active learning is perfomed.

    Args:
        actor_critic (_type_): _description_
        agent (_type_): _description_
        args (_type_): _description_
        num_updates (_type_): _description_
        envs (_type_): _description_
        device (_type_): _description_
    """
    global total_num_steps_progess

    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(args.num_env_steps_progress) // args.num_steps // args.num_processes

    for steps in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                active_agent.optimizer, steps, num_updates,
                active_agent.optimizer.lr if args.algo == "acktr" else args.lr)
            
            
        # nmb of steps (rollouts) before update
        for step in range(args.num_steps):
            
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = big_policy.act(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            print(obs)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)
        

        with torch.no_grad():
            next_value = big_policy.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        # gradient update
        value_loss, action_loss, dist_entropy = active_agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (steps % args.save_interval == 0 or steps == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic_active.state_dict(),
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, env_name + f"-{steps}" + f"-active-visit{vis}" + ".pt"))

        
        # log metrics
        total_num_steps_progess.setdefault(env_name, 0)
        total_num_steps_progess[env_name] += args.num_processes * args.num_steps
        if steps % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(steps, total_num_steps_progess[env_name],
                        int(total_num_steps_progess[env_name] / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            wandb.log({f"Progress/Training/Score-{env_name}": np.mean(episode_rewards),
                       f"Progress/Training/value_loss-{env_name}": value_loss,
                       f"Progress/Training/action_loss-{env_name}": action_loss,
                       f"Progress/Training/dist_entropy-{env_name}": dist_entropy,
                       f"Progress/Training/Timesteps-{env_name}": total_num_steps_progess[env_name]})
            
            
        # # eval during phase or not
        # if (args.eval_interval is not None and len(episode_rewards) > 1 and steps % args.eval_interval == 0):
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     # evaluate(actor_critic, obs_rms, args.env_name, args.seed,
        #     #          args.num_processes, eval_log_dir, device)


def compress(big_policy, kb_agent, actor_critic_kb, ewc, args, envs, device, env_name, eval_log_dir, vis, agn=False):
    global total_num_steps_compress
    global environements

    rollouts = RolloutStorage(args.num_steps, args.num_processes,envs.observation_space.shape, envs.action_space)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    if agn:
        num_updates = int(args.num_env_steps_agnostic) // args.num_steps // args.num_processes
    else:
        num_updates = int(args.num_env_steps_compress) // args.num_steps // args.num_processes
    
    if ewc is not None: ewc.ewc_timestep_counter = 0 # restart counter

    if agn and ewc is not None:
        ewc.ewc_start_timestep = int(args.num_env_steps_agnostic)//2
    if agn == False and ewc is not None:
        ewc.ewc_start_timestep = args.ewc_start_timestep_after
    else: 
        pass

    for steps in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                kb_agent.optimizer, steps, num_updates,
                kb_agent.optimizer.lr if args.algo == "acktr" else args.lr)
            
        # nmb of steps (rollouts) before update
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = big_policy.act(rollouts.obs[step])
                
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)
        

        with torch.no_grad():
            next_value = big_policy.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        # update counter #tdb (should it be applied on the global timestep or the current one?)
        if ewc is not None:
            ewc.ewc_timestep_counter += args.num_processes * args.num_steps

        # gradient update
        total_loss, kl_loss, ewc_loss = kb_agent.update(rollouts, ewc)
        print(f"Total loss: {total_loss}, KL loss: {kl_loss}, EWC loss: {ewc_loss}")
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (steps % args.save_interval == 0 or steps == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic_kb.state_dict(),
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, env_name + f"-{steps}" + f"-kb-visit{vis}" + ".pt"))

        # log metrics
        total_num_steps_compress.setdefault(env_name, 0)
        total_num_steps_compress[env_name] += args.num_processes * args.num_steps
        if steps % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(steps, total_num_steps_compress[env_name],
                        int(total_num_steps_compress[env_name] / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            wandb.log({f"Compress/Training/Score-{env_name}": np.mean(episode_rewards),
                       f"Compress/Training/KL-Loss-{env_name}": np.mean(kl_loss),
                       f"Compress/Training/total-loss-{env_name}": np.mean(total_loss),
                       f"Compress/Training/ewc_loss-{env_name}": np.mean(ewc_loss),
                       f"Compress/Training/Timesteps-{env_name}": total_num_steps_compress[env_name]})
        
        if (args.eval_interval is not None and (steps+1) % args.eval_interval == 0):
            print(5*"#", "Evaluation phase", 5*"#")
            for eval_env_name in environements:
                print(2*"#", f"Eval {eval_env_name}", 2*"#")
                evaluate(args, actor_critic_kb, eval_env_name, args.seed, args.num_processes, eval_log_dir, device)

def agnostic(big_policy, active_agent, forward_model, args, envs, device, env_name, vis):
    """The progress phase, where the active learning is perfomed.

    Args:
        actor_critic (_type_): _description_
        agent (_type_): _description_
        args (_type_): _description_
        num_updates (_type_): _description_
        envs (_type_): _description_
        device (_type_): _description_
    """
    global total_num_steps_agnostic

    rollouts = RolloutStorage(args.num_steps, args.num_processes,envs.observation_space.shape, envs.action_space)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_steps_list = deque(maxlen=100)
    episode_intrinsic_rewards = deque(maxlen=100)

    # plot
    episode_intrinsic_rewards = []
    steps_list = []

    start = time.time()
    num_updates = int(args.num_env_steps_agnostic) // args.num_steps // args.num_processes
    fwd_criterion = nn.MSELoss()
    episode_steps = 0

    for steps in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                active_agent.optimizer, steps, num_updates,
                active_agent.optimizer.lr if args.algo == "acktr" else args.lr)
            
        # nmb of steps (rollouts) before update
        fwd_losses = []
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = big_policy.act(rollouts.obs[step])

            episode_steps += 1

            # Obser reward and next obs
            new_obs, reward, done, infos = envs.step(action)

            # One-hot encode the action
            # print(action)
            num_actions = 4
            action_oh = torch.zeros(action.shape[0], num_actions).to(device)  # Shape will be [8, 18]
            action_oh.scatter_(1, action, 1)  # One-hot encoding
            # print(action_oh)
            pred_phi, phi = forward_model(rollouts.obs[step].clone(),  new_obs.clone(), action_oh) # forward model forward

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_steps_list.append(episode_steps)
                    episode_steps = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            # Calculate forward prediction error
            # fwd_loss = fwd_criterion(pred_phi, phi)
            fwd_loss = torch.norm(pred_phi - phi, dim=-1, p=2, keepdim=True)
            # print(fwd_loss, fwd_loss.shape)

            # Calculate intrinsic reward
            intrinsic_reward = torch.log(fwd_loss+1).detach().cpu()
            reward = intrinsic_reward
            # print(intrinsic_reward.shape)
            # intrinsic_reward = 0.2 * fwd_loss.detach().to("cpu")
            # intrinsic_reward = ((pred_phi-phi).pow(2)).mean(dim=1).detach().view(-1, 1).cpu()
            # print(intrinsic_reward.sum(dim=0))
            # print(reward, reward.shape)

            # Append the current step to the steps list
            steps_list.append(episode_intrinsic_rewards)

            episode_intrinsic_rewards.append(intrinsic_reward.mean().item())
            fwd_losses.append(fwd_loss)
            rollouts.insert(new_obs, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = big_policy.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        # gradient update
        value_loss, action_loss, dist_entropy = active_agent.update(rollouts, fwd_losses=fwd_losses)
        rollouts.after_update()
        
        # log metrics
        total_num_steps_agnostic.setdefault(env_name, 0)
        total_num_steps_agnostic[env_name] += args.num_processes * args.num_steps
        if steps % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean intr.reward {:.8f}\n"
                .format(steps, total_num_steps_agnostic[env_name],
                        int(total_num_steps_agnostic[env_name] / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_intrinsic_rewards)))
            wandb.log({f"Agnostic/Training/Score-{env_name}": np.mean(episode_rewards),
                       f"Agnostic/Training/value_loss-{env_name}": value_loss,
                       f"Agnostic/Training/action_loss-{env_name}": action_loss,
                       f"Agnostic/Training/dist_entropy-{env_name}": dist_entropy,
                       f"Agnostic/Training/Timesteps-{env_name}": total_num_steps_agnostic[env_name],
                       f"Agnostic/Training/IntrinsicReward-{env_name}": np.mean(episode_intrinsic_rewards),
                       f"Agnostic/Training/EpisodeSteps_per_episode-{env_name}": np.mean(episode_steps_list)})
