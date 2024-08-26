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

from src import algo, utils, progress, compress, agnostic, progress_progressive
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
from src.model_pnn.ColumnGenerator import Column_generator_CNN
from src.model_pnn.ProgNet import ProgNet

import multiprocessing, logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# logger = multiprocessing.log_to_stderr()
# logger.setLevel(multiprocessing.SUBDEBUG)

# total timesteps across all visits
# total_num_steps_progess = {}
# total_num_steps_compress = {}
# total_num_steps_agnostic = {}
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
    # torch.set_num_threads(8)

    # fallback for cpu
    if args.device == "mps":
        device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logging.info(f'Model runs on backend: {device}')
    
    # set up the environment
    environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "AirRaidNoFrameskip-v4"]
    agnostic_environements = ["SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    test_environements = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "AirRaidNoFrameskip-v4"]

    #### init first environment for architecture initialization ####
    envs = make_vec_envs(args.env_name, args.seed, 1, args.gamma, args.log_dir, device, False)

    #### init policies ####
    print("action shape", envs.action_space)

    actor_critic_active = Policy(
            envs.observation_space.shape,
            envs.action_space)
    actor_critic_active.to(device)

    actor_critic_kb = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic_kb.to(device)
    
    adaptor = Adaptor()
    adaptor.to(device)

    if args.algo.split("/")[0] == "progress-compress" or args.algo.split("/")[0] == "ewc-online":
        big_policy = BigPolicy(actor_critic_kb, actor_critic_active, adaptor)
        forward_model = IntrinsicCuriosityModule(envs.observation_space.shape, envs.action_space.n)
        forward_model.to(device)
        
        agnostic_flag = args.agnostic_phase
        samples_nmb = args.agn_samples # generate x samples from the distribution (visiting rooms)

    elif args.algo.split("/")[0] == "progressive-nets":
        column_generator = Column_generator_CNN(num_of_conv_layers=2, kernel_size=8, num_of_classes=1, num_dens_Layer=1)
        big_policy = ProgNet(column_generator, output_size=envs.action_space.n)
        idx = big_policy.addColumn(device=device) # add column to the network for each task
        logging.info(f"===== New column generated: {idx} =====")
        forward_model = None

    
    #### setups gradient updates, i.e. a2c and knowledge distill ####
    active_agent = algo.A2C(
        big_policy,
        actor_critic_active,
        args.value_loss_coef,
        args.entropy_coef,
        forward_model=forward_model,
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
                steps_calculate_fisher=args.steps_calculate_fisher)


    if args.algo.split("/")[0] == "progress-compress":
        logging.info("Progress and Compress Framework Algorithm")
        big_policy.train()
        progress_and_compress(actor_critic_active, actor_critic_kb, big_policy, adaptor, forward_model, active_agent, kb_agent, ewc, args, device, eval_log_dir, agnostic_flag, agnostic_environements, test_environements, samples_nmb, environements)
    elif args.algo.split("/")[0] == "ewc-online":
        logging.info("EWC-Online Algorithm")
        big_policy.train()
        ewc_online(args, big_policy, active_agent, actor_critic_active, actor_critic_kb, envs, device, ewc, test_environements, eval_log_dir, environements)
    elif args.algo.split("/")[0] == "progressive-nets":
        logging.info("Progressive Neural Network Algorithm")
        big_policy.train()
        progressive_nets(args, big_policy, active_agent, envs, device, test_environements, eval_log_dir, environements)

    logging.info("Training done!")


def progressive_nets(args, big_policy, active_agent, envs, device, test_environements, eval_log_dir, environements):
    """Training process of the progressive neural network.
    """
    for vis in range(args.visits):
        for task_id, env_name in enumerate(environements):
            logging.info(f"{20 * '#'} Visit {vis + 1} of {env_name} {20 * '#'}")
            envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

            # unfreeze column for the specific task
            big_policy.unfreezeColumn(task_id)
            progress_progressive(policy=big_policy, a2c_agent=active_agent, args=args, envs=envs, device=device, env_name=env_name, vis=vis, task_id=task_id)
            big_policy.freezeAllColumns()

            # evaluation of one column for assessing forward transfer.
            logging.info(f"{5 * '#'} Evaluation phase {5 * '#'}")
            for eval_task_id, eval_env_name in enumerate(test_environements):
                logging.info(f"{5 * '#'} Evaluate: {eval_env_name} {5 * '#'}")
                if eval_task_id <= task_id:
                    evaluate(args, big_policy, eval_env_name, args.seed, args.num_processes, eval_log_dir, device, "progressive-nets", eval_task_id)
            big_policy.train()

            # add next columns for subsequent tasks
            if vis == 0 and task_id < len(environements) - 1:
                idx = big_policy.addColumn(device=device) # add a column to the network for each task
                active_agent.optimizer.add_param_group({'params': big_policy.getColumn(idx).parameters()}) # add the new column to the optimizer
                logging.info("Optimizer param groups:")
                for param_group in active_agent.optimizer.param_groups:
                    for param in param_group['params']:
                        for name, param_in_model in big_policy.named_parameters():
                            if param_in_model is param:
                                print(f"Optimizing parameter: {name}")
                logging.info(f"===== New column generated: {idx} =====")

def ewc_online(args, big_policy, active_agent, actor_critic_active, actor_critic_kb, envs, device, ewc, test_environements, eval_log_dir, environements):
    """The "progress" method is just a forward pass and backward pass of the active policy, i.e. one NN module.
    """
    actor_critic_active.reset_weights()
    for vis in range(args.visits):
        for env_name in environements:
            logging.info(f"{20 * '#'} Visit {vis + 1} of {env_name} {20 * '#'}")
            envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)
            
            freeze_everything(actor_critic_kb)
            unfreeze_everything(big_policy)

            # as the progress phase is just a forward pass and backward pass of the active policy, i.e. one NN module, we can use the progress method here for ewc-online.
            if big_policy.experience > 0:
                progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name, vis, ewc)
            else:
                progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name, vis, None)
                big_policy.experience += 1

            # compute the fisher
            ewc.update_parameters(actor_critic_active, env_name) # also like init function in this context
            gather_fisher_samples(actor_critic_active, ewc, args, envs, device)

            # evaluation of one column for assessing forward transfer.
            logging.info(f"{5 * '#'} Evaluation phase {5 * '#'}")
            for eval_env_name in test_environements:
                logging.info(f"{5 * '#'} Evaluate: {eval_env_name} {5 * '#'}")
                evaluate(args, big_policy, eval_env_name, args.seed, args.num_processes, eval_log_dir, device, "ewc-online")
            big_policy.train()

def progress_and_compress(actor_critic_active, actor_critic_kb, big_policy, adaptor, forward_model, active_agent, kb_agent, ewc, args, device, eval_log_dir, agnostic_flag, agnostic_environements, test_environements, samples_nmb, environements):
    for vis in range(args.visits):
        # start agnostic phase before pandc
        if agnostic_flag != False:
            print(5*"#", "Agnostic phase", 5*"#")

            agnostic_flag = False # set the flag to false, because the agnostic phase is only activatet once
            
            # reset weights
            actor_critic_active.reset_weights()
            adaptor.reset_weights()

            for j in range(samples_nmb):

                sample = np.random.choice(agnostic_environements, 1) # sample name out of distribution
                print(f"SampledEnv: {j}: {sample[0]}")
                agn_environments = make_vec_envs(sample[0], args.seed, args.num_processes, args.gamma, args.log_dir, device, False) # create env

                unfreeze_everything(big_policy) # unfreeze the big policy (here upate the active policy)
                freeze_everything(big_policy.policy_a) # otherwise the policy_a would be updated in the progress phase
                freeze_everything(actor_critic_kb)
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
                    compress(big_policy, kb_agent, actor_critic_kb, ewc, args, agn_environments, device, sample[0], eval_log_dir, vis, agn=True)
                    big_policy.experience += 1
                else:
                    compress(big_policy, kb_agent, actor_critic_kb, None, args, agn_environments, device, sample[0], eval_log_dir, vis, agn=True)
                    big_policy.experience += 1

                # compute the fisher
                ewc.update_parameters(actor_critic_kb, "agnostic") # also like init function in this context
                gather_fisher_samples(actor_critic_kb, ewc, args, agn_environments, device)

                # update the model, to not include into the comp hist in the compress phase
                # print("Before update:", big_policy.policy_a.state_dict())
                big_policy.update_model(actor_critic_kb)
                # print("After update:", big_policy.policy_a.state_dict())


                # use lateral connection after training min of one task, i.e. right after the above code
                big_policy.use_lateral_connection = True


        logging.info(f"{5 * '#'} PandC Framework {5 * '#'}")
        for env_name in environements:

            # reset weights
            big_policy.policy_b.reset_weights()

            logging.info(f"{20 * '#'} Visit {vis + 1} of {env_name} {20 * '#'}")
            # init environment
            envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

            # progress phase
            logging.info(f"{5 * '#'} Progress phase {5 * '#'}")
            unfreeze_everything(big_policy) # unfreeze the big policy (here upate the active policy)
            freeze_everything(big_policy.policy_a) # otherwise the policy_a would be updated in the progress phase
            freeze_everything(actor_critic_kb)
            progress(big_policy, active_agent, actor_critic_active, args, envs, device, env_name, vis)

            # compress phase
            logging.info(f"{5 * '#'} Compress phase {5 * '#'}")
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
            logging.info(f"{5 * '#'} Evaluation phase {5 * '#'}")
            for eval_env_name in test_environements:
                logging.info(f"{5 * '#'} Evaluate: {eval_env_name} {5 * '#'}")
                evaluate(args, big_policy, eval_env_name, args.seed, args.num_processes, eval_log_dir, device, "active") # active columns is embedded into big policy
                evaluate(args, actor_critic_kb, eval_env_name, args.seed, args.num_processes, eval_log_dir, device, "kb")
            big_policy.train()

            # update the model, to not include into the comp hist in the compress phase
            # print("Before update:", big_policy.policy_a.state_dict())
            big_policy.update_model(actor_critic_kb)
            # print("After update:", big_policy.policy_a.state_dict())

            # use lateral connection after training min of one task, i.e. right after the above code
            big_policy.use_lateral_connection = True

if __name__ == "__main__":
    main()
