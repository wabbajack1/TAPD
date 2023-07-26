"""Some code is from https://github.com/GMvandeVen/continual-learning/blob/ff0e03cb913ac0dea4fc59058968b1e6784decfd/utils.py.
This module should help for debugging and other purposes of a ANN.
"""
from torch import nn
from commons.EWC import EWC
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import common.wrappers as wrappers
import time
import torch
import gym
import wandb
import numpy as np

################################
## global counters            ##
################################
frame_number_eval = {}
steps = {}


################################
## Model-inspection functions ##
################################
def evaluate(model, env_name, save_dir, num_episodes, seed):
    
    # collect frame across all visits
    global frame_number_eval
    global steps
    
    # init env but do not clip the rewards in eval phase                    
    env = environment_wrapper(save_dir, env_name=env_name, video_record=True, clip_rewards=False)
    
    evaluation_scores = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_reward_orignal = 0

        while not env.was_real_done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model.act(state_tensor.to(model.device))
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if env_name not in steps:
                steps[env_name] = 0
            else:
                steps[env_name] += 1
                
            if env.was_real_done:
                if env_name not in frame_number_eval:
                    frame_number_eval[env_name] = 0
                    frame_number_eval[env_name] += info["episode_frame_number"]
                else:
                    frame_number_eval[env_name] += info["episode_frame_number"]
                    
                wandb.log({f"Evaluation score-{env_name}-{model.__class__.__name__}": episode_reward, f"Frame-# Evaluation-{env_name}": frame_number_eval[env_name], f"Steps-# Evaluation-{env_name}":steps[env_name]})
                
        evaluation_scores.append(episode_reward)

    return np.mean(evaluation_scores) #, np.mean(evaluation_scores_original)

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params

def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("Model-name: \"" + model.name + "\"")
    print(40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-" + "\n\n")

def test_disitillation(visits, agent, environments, max_steps_progress, max_steps_compress, save_dir, evaluation_epsiode, batch_size_fisher, batch_number_fisher, ewcgamma, ewclambda, seed):
    
    # see eval before distillation
    print(20*"=", "Evaluation started", 20*"=")
    for env_name_eval in ["PongNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]:
        evaluation_data = evaluate(model=agent.kb_model, env_name=env_name_eval, save_dir=save_dir, num_episodes=evaluation_epsiode, seed=seed)
        avg_score = evaluation_data
        print(f"Steps: {steps[env_name_eval]}, Frames in {env_name_eval}: {frame_number_eval[env_name_eval]}, Avg Evaluation score: {avg_score}")
    print(20*"=","Evaluation completed", 20*"=")
    
    # compute fisher for pong
    print("############## Creating workers ##############")
    agent.reinitialize_workers("PongNoFrameskip-v4")
            
    for k in range(batch_number_fisher):
        with ThreadPoolExecutor(max_workers=len(agent.workers)) as executor:
            # Submit tasks to the executor and collect results based on the current policy of kb knowledge
            futures = [executor.submit(agent.collect_batch, worker, "Compress", batch_size_fisher) for worker in agent.workers]
            batches = [f.result() for f in as_completed(futures)]
        
        # print(len(batches))
        for j, (states, actions, true_values) in enumerate(batches):
            # print(len(states), states[0].shape)
            for i, _ in enumerate(states):
                agent.memory.push(
                    states[i],
                    actions[i],
                    true_values[i]
                )
        
    if agent.ewc_init:
        # take the latest env and calculate the fisher
        ewc = EWC(agent=agent, model=agent.kb_model, ewc_lambda=ewclambda, ewc_gamma=ewcgamma, batch_size_fisher=batch_size_fisher, device=agent.device, env_name="PongNoFrameskip-v4")
        agent.ewc_init = False
    # else: # else running calulaction taken the last fisher into consideration
    #     ewc.update(agent, agent.kb_model, env_name) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
    
    # distill knowledge from beamrider and protect pong
    print("Distillation + EWC")
    print("############## Creating workers ##############")
    agent.reinitialize_workers("BeamRiderNoFrameskip-v4")
    agent.compress_training(max_steps_compress, ewc)
    
    # see eval after distillation
    print(20*"=", "Evaluation started", 20*"=")
    for env_name_eval in ["PongNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]:
        evaluation_data = evaluate(model=agent.kb_model, env_name=env_name_eval, save_dir=save_dir, num_episodes=evaluation_epsiode, seed=seed)
        avg_score = evaluation_data
        print(f"Steps: {steps[env_name_eval]}, Frames in {env_name_eval}: {frame_number_eval[env_name_eval]}, Avg Evaluation score: {avg_score}")
    print(20*"=","Evaluation completed", 20*"=")
            
    pass

def environment_wrapper(save_dir, env_name, video_record=False, clip_rewards=True):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """

    
    env = wrappers.make_atari(env_name, full_action_space=True)
    if video_record:
        path = (save_dir / "video" / f"{env.spec.id}_{time.time()}")
        env = gym.wrappers.Monitor(env, path, mode="evaluation")
    env = wrappers.wrap_deepmind(env, scale=True, clip_rewards=clip_rewards) 
    env = wrappers.wrap_pytorch(env)
    return env

class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr