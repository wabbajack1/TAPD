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

from new_a2c import algo, utils
from new_a2c.algo import gail
from new_a2c.arguments import get_args
from new_a2c.envs import make_vec_envs
from new_a2c.model import Policy, BigPolicy, Adaptor
from new_a2c.storage import RolloutStorage
from evaluation import evaluate
import wandb
from new_a2c.utils import freeze_everything, unfreeze_everything

# total timesteps across all visits
total_num_steps_progess = {}
total_num_steps_compress = {}


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("mps:0" if args.cuda else "cpu")

    environements = ["PongNoFrameskip-v4"]

    #### init first environment for architecture initialization ####
    envs = make_vec_envs(args.env_name, args.seed, 1, args.gamma, args.log_dir, device, False)

    #### init policies ####
    actor_critic_active = Policy(
        envs.observation_space.shape,
        envs.action_space)
    # actor_critic_active.load_state_dict(torch.load("/Users/kerekmen/Developer/agnostic_rl/trained_models/a2c/active.pt")[0])
    actor_critic_active.to(device)

    actor_critic_kb = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic_kb.to(device)

    adaptor = Adaptor()
    adaptor.to(device)

    big_policy = BigPolicy(actor_critic_kb, actor_critic_active, adaptor)


    #### setups losses, i.e. a2c and knowledge distill ####
    active_agent = algo.A2C_ACKTR(
        big_policy,
        actor_critic_active,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm)
    
    kb_agent = algo.Distillation(
        big_policy,
        actor_critic_kb,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm)
    
    # ewc = algo.EWC()
    
    for vis in range(args.visits):
        for env_name in environements:
            print(20*"#", f"Visit {vis + 1} of {env_name}", 20*"#")
            # init environment
            envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

            # progress phase
            print(5*"#", "Progress phase", 5*"#")
            freeze_everything(actor_critic_kb)
            unfreeze_everything(actor_critic_active)
            unfreeze_everything(adaptor)
            progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name)

            # compress phase
            print(5*"#", "Compress phase", 5*"#")
            freeze_everything(actor_critic_active)
            freeze_everything(adaptor)
            unfreeze_everything(actor_critic_kb) # different from big_policy.policy_b in the memory
            compress(actor_critic_active, kb_agent, actor_critic_kb, args, envs, device, env_name, eval_log_dir)

            # calculate ewc-online to include for next compress activity, i.e. after compressing each task, update EWC
            # collect samples from current policy (here re-run the last x steps of progress)
            # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (becaue on-policy i.e. without replay buffer)
            # collect training data from current kb knowledge policy
            # if agent.ewc_init == True:
            #     print("Distillation")
            #     agent.compress_training(max_steps_compress, None)
            # else:
            #     print("Distillation + EWC")
            #     agent.compress_training(max_steps_compress, ewc)

            # evaluation of kb column
            print(5*"#", "Evaluation phase", 5*"#")
            obs_rms = utils.get_vec_normalize(envs)
            evaluate(args, actor_critic_kb, obs_rms, env_name, args.seed, args.num_processes, eval_log_dir, device)

            # update the model, to not include into the comp hist in the compress phase
            big_policy.update_model(actor_critic_kb)

            # re-init weights of active column
            actor_critic_active.reset_weights()
            adaptor.reset_weights()

            # use lateral connection after training min of one task, i.e. right after the above code
            big_policy.use_lateral_connection = True

    print(20*"#", f"Training done!", 20*"#")

def progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name):
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

    rollouts = RolloutStorage(args.num_steps, args.num_processes,envs.observation_space.shape, envs.action_space)
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
            ], os.path.join(save_path, "active" + ".pt"))

        
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

def compress(big_policy, kb_agent, actor_critic_kb, args, envs, device, env_name, eval_log_dir):
    global total_num_steps_compress

    rollouts = RolloutStorage(args.num_steps, args.num_processes,envs.observation_space.shape, envs.action_space)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(args.num_env_steps_compress) // args.num_steps // args.num_processes

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

        # gradient update
        kl_loss = kb_agent.update(rollouts)
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
            ], os.path.join(save_path, "kb" + ".pt"))

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
                       f"Compress/Training/Timesteps-{env_name}": total_num_steps_compress[env_name]})
        
        # if (args.eval_interval is not None and len(episode_rewards) > 1 and steps % args.eval_interval == 0):
        #     print(5*"#", "Evaluation phase", 5*"#")
        #     obs_rms = utils.get_vec_normalize(envs)
        #     evaluate(args, actor_critic_kb, obs_rms, env_name, args.seed, args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="A2C-ikostrikov",
        entity="agnostic",
        # mode="disabled",
        # id="nd07r8xn",
        # resume="allow"
    )
    main()

    # args = get_args()
    
    # log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    # torch.set_num_threads(1)
    # device = torch.device("mps:0" if args.cuda else "cpu")

    # # init first environment
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

    # actor_critic_active = Policy(
    #     envs.observation_space.shape,
    #     envs.action_space)
    # actor_critic_active.to(device)

    # actor_critic_kb = Policy(
    #     envs.observation_space.shape,
    #     envs.action_space)
    # actor_critic_kb.to(device)

    # big_net = BigPolicy(actor_critic_kb, actor_critic_active)

    # for n, p in big_net.named_parameters():
    #     print(n)