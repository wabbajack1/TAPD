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

from src import algo, utils, progress, compress, agnostic
from src.arguments import get_args
from src.envs import make_vec_envs
from src.model import Policy, BigPolicy, Adaptor, IntrinsicCuriosityModule
from src.storage import RolloutStorage
from evaluation import evaluate
import wandb
from src.utils import freeze_everything, unfreeze_everything
from datetime import datetime
from src.algo import gather_fisher_samples
import yaml

import multiprocessing, logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# logger = multiprocessing.log_to_stderr()
# logger.setLevel(multiprocessing.SUBDEBUG)

# total timesteps across all visits
total_num_steps_progess = {}
total_num_steps_compress = {}
total_num_steps_agnostic = {}
environements = []


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    global environements
    args = get_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="Progress and Compress - Prediction",
        entity="agnostic",
        config=args,
        mode="online" if args.log_wandb else "disabled",
    )

    # set seed and log dir
    torch.manual_seed(args.seed)
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.algo = args.algo + "/" + timestamp_str
    torch.set_num_threads(1)
    device = torch.device("mps:0" if args.mps else "cpu")
    logging.info(f'Model runs on backend: {device}')
    
    # set up the environment
    environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    agnostic_environements = ["SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    test_environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]

    #### init first environment for architecture initialization ####
    envs = make_vec_envs(args.env_name, args.seed, 1, args.gamma, args.log_dir, device, False)

    #### init policies ####
    actor_critic_active = Policy(
        envs.observation_space.shape,
        envs.action_space)
    # actor_critic_active.load_state_dict(torch.load(".pt"))
    actor_critic_active.to(device)

    actor_critic_kb = Policy(
        envs.observation_space.shape,
        envs.action_space)
    # actor_critic_kb.load_state_dict(torch.load(".pt"))
    actor_critic_kb.to(device)

    adaptor = Adaptor()
    adaptor.to(device)

    big_policy = BigPolicy(actor_critic_kb, actor_critic_active, adaptor)

    forward_model = IntrinsicCuriosityModule(envs.observation_space.shape, envs.action_space.n)
    forward_model.to(device)

    #### setups losses, i.e. a2c and knowledge distill ####
    active_agent = algo.A2C(
        big_policy,
        actor_critic_active,
        forward_model,
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
    
    ewc = algo.EWConline(args.entropy_coef,
                   args.ewc_lambda,
                   args.ewc_gamma,
                   args.ewc_start_timestep_after,
                   max_grad_norm=args.max_grad_norm,
                   steps_calucate_fisher=args.steps_calucate_fisher)
    
    
    agnostic_flag = args.agnostic_phase
    samples_nmb = args.agn_samples # generate x samples from the distribution (visiting rooms)

    big_policy.train()
    for vis in range(args.visits):
        # start agnostic phase before pandc
        if agnostic_flag != False:
            print(5*"#", "Agnostic phase", 5*"#")
            agnostic_flag = False # set the flag to false, because the agnostic phase is only activatet once

            for j in range(samples_nmb):
                # reset weights
                actor_critic_active.reset_weights()
                adaptor.reset_weights()
                
                sample = np.random.choice(agnostic_environements, 1) # sample name out of distribution
                print(f"SampledEnv: {j}: {sample[0]}")
                agn_environments = make_vec_envs(sample[0], args.seed, args.num_processes, args.gamma, args.log_dir, device, False) # create env

                freeze_everything(actor_critic_kb)
                unfreeze_everything(big_policy)
                agnostic(big_policy, active_agent, forward_model, args, agn_environments, device, sample[0], vis)

                # compress phase
                print(5*"#", "Compress phase", 5*"#")
                freeze_everything(big_policy)
                unfreeze_everything(actor_critic_kb) # different from big_policy.policy_b in the memory

                # calculate ewc-online to include for next compress activity, i.e. after compressing each task, update EWC
                # collect samples from current policy (here re-run the last x steps of progress)
                # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (becaue on-policy i.e. without replay buffer)
                # collect training data from current kb knowledge policy
                # for j in range(samples_nmb):
                #     sample = np.random.choice(environements, 1, p=[1/3, 1/3, 1/3])
                #     envs = make_vec_envs(sample[0], args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
                if big_policy.experience > 0:
                    print("Distill + EWC")
                    compress(big_policy, kb_agent, actor_critic_kb, ewc, args, agn_environments, device, sample[0], eval_log_dir, vis, agn=False)
                    big_policy.experience += 1
                else:
                    compress(big_policy, kb_agent, actor_critic_kb, None, args, agn_environments, device, sample[0], eval_log_dir, vis, agn=False)
                    big_policy.experience += 1

                # compute the fisher
                ewc.update_parameters(actor_critic_kb, "agnostic") # also like init function in this context

                # for env_name in environements:
                #     envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
                gather_fisher_samples(actor_critic_kb, ewc, args, agn_environments, device)

                # update the model, to not include into the comp hist in the compress phase
                print("Before update:", big_policy.policy_a.state_dict())
                big_policy.update_model(actor_critic_kb)
                print("After update:", big_policy.policy_a.state_dict())


                # use lateral connection after training min of one task, i.e. right after the above code
                big_policy.use_lateral_connection = True

    
        print(5*"#", "PandC Framework", 5*"#")
        for env_name in environements:

            # reset weights
            actor_critic_active.reset_weights()
            adaptor.reset_weights()


            print(20*"#", f"Visit {vis + 1} of {env_name}", 20*"#")
            # init environment
            envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
            # envs = utils.get_vec_normalize(envs)

            # progress phase
            print(5*"#", "Progress phase", 5*"#")
            freeze_everything(actor_critic_kb)
            unfreeze_everything(big_policy)
            progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name, vis)

            # compress phase
            print(5*"#", "Compress phase", 5*"#")
            freeze_everything(big_policy)
            unfreeze_everything(actor_critic_kb) # different from big_policy.policy_b in the memory

            # calculate ewc-online to include for next compress activity, i.e. after compressing each task, update EWC
            # collect samples from current policy (here re-run the last x steps of progress)
            # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (becaue on-policy i.e. without replay buffer)
            # collect training data from current kb knowledge policy
            if big_policy.experience > 0:
                print("Distill + EWC")
                compress(big_policy, kb_agent, actor_critic_kb, ewc, args, envs, device, env_name, eval_log_dir, vis)
                big_policy.experience += 1
            else:
                compress(big_policy, kb_agent, actor_critic_kb, None, args, envs, device, env_name, eval_log_dir, vis)
                big_policy.experience += 1

            # compute the fisher
            ewc.update_parameters(actor_critic_kb, env_name) # also like init function in this context
            gather_fisher_samples(actor_critic_kb, ewc, args, envs, device)

            # evaluation of active column for assessing forward transfer
            print(5*"#", "Evaluation phase", 5*"#")
            for eval_env_name in test_environements:
                print(2*"#", f"Eval {eval_env_name}", 2*"#")
                evaluate(args, actor_critic_kb, eval_env_name, args.seed, args.num_processes, eval_log_dir, device)

            # update the model, to not include into the comp hist in the compress phase
            # print("Before update:", big_policy.policy_a.state_dict())
            big_policy.update_model(actor_critic_kb)
            # print("After update:", big_policy.policy_a.state_dict())

            # use lateral connection after training min of one task, i.e. right after the above code
            big_policy.use_lateral_connection = True

    print(20*"#", f"Training done!", 20*"#")

if __name__ == "__main__":
    main()
