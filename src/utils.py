"""Some code is from https://github.com/GMvandeVen/continual-learning/blob/ff0e03cb913ac0dea4fc59058968b1e6784decfd/utils.py.
This module should help for debugging and other purposes of a ANN.
"""
from torch import nn
from commons.EWC import EWC
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor, as_completed
import os
import envs.wrappers as wrappers
import time
import torch
import gym
import wandb
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers import AtariPreprocessing, FrameStack
from envs.wrappers import ClipRewardEnv, EpisodicLifeEnv
from unified_action_space import UnifiedActionWrapper

################################
## global counters            ##
################################
frame_number_eval = {}
steps = {}


################################
## Model-inspection functions ##
################################
def evaluate(model, env_name, num_steps, seed=None):
    
    # collect frame and steps across all visits
    global frame_number_eval
    global steps
    
    # init env but do not clip the rewards in eval phase                    
    env = environment_wrapper(env_name=env_name, mode="rgb_array",clip_rewards=False)
    env.seed(seed=seed)
    
    evaluation_scores = []
    episode_reward = 0
    state = env.reset()
    for k in range(num_steps):
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        action = model.act(state_tensor.to(model.device))
        next_state, reward, done, _, info = env.step(action)
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
                
            wandb.log({f"Evaluation/Score_{env_name}-{model.__class__.__name__}": episode_reward, 
                        f"Evaluation/Frame-#-{env_name}": frame_number_eval[env_name], 
                        f"Evaluation/Steps-#-{env_name}":steps[env_name]})

            print(k, episode_reward, env.was_real_done, steps[env_name])
            evaluation_scores.append(episode_reward)
            episode_reward = 0 
            state = env.reset()

    wandb.log({f"Mean over {num_steps} evaluation steps for {env_name}-{model.__class__.__name__}": np.mean(evaluation_scores[-100:])})
    print(f"Mean over {num_steps} evaluation steps for {env_name}-{model.__class__.__name__}", np.mean(evaluation_scores[-100:]))
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

def test_disitillation(agent, max_steps_compress, save_dir, evaluation_epsiode, batch_size_fisher, batch_number_fisher, ewcgamma, ewclambda, ewc_start, seed, eval_after_compress=True):
    
    # see eval before distillation
    _evaluate(agent, save_dir, evaluation_epsiode, seed=seed)
    
    # compute fisher for pong
    print(10*"#", " Collect rollout for EWC ", 10*"#")
    print("############## Creating workers ##############")
    agent.reinitialize_workers("PongNoFrameskip-v4")
    for k in range(batch_number_fisher):
        # with ThreadPoolExecutor(max_workers=len(agent.workers)) as executor:
        # Submit tasks to the executor and collect results based on the current policy of kb knowledge
        # futures = [executor.submit(agent.collect_batch, worker, "Compress", batch_size_fisher) for worker in agent.workers]
        # batches = [f.result() for f in as_completed(futures)]

        for worker in agent.workers:
            states, actions, true_values = agent.collect_batch(worker, "Compress", batch_size_fisher)
            for i, _ in enumerate(states):
                agent.memory.push(
                    states[i],
                    actions[i],
                    true_values[i]
                )
        
    if agent.ewc_init:
        # take the latest env and calculate the fisher
        ewc = EWC(agent=agent, model=agent.kb_model, ewc_lambda=ewclambda, ewc_gamma=ewcgamma, batch_size_fisher=batch_size_fisher, device=agent.device, ewc_start_timestep_after=ewc_start, env_name="PongNoFrameskip-v4")
        agent.ewc_init = False
    # else: # else running calulaction taken the last fisher into consideration
    #     ewc.update(agent, agent.kb_model, env_name) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
    
    # distill knowledge from beamrider and protect pong
    print("Distillation + EWC")
    print("############## Creating workers ##############")
    agent.reinitialize_workers("BeamRiderNoFrameskip-v4")

    if eval_after_compress == False:
        until_steps = 100000
        for _ in range(0, max_steps_compress, until_steps):
            agent.compress_training(until_steps, ewc)
            _evaluate(agent, save_dir, evaluation_epsiode, seed=seed)
    else:
        agent.compress_training(max_steps_compress, ewc)
        _evaluate(agent, save_dir, evaluation_epsiode, seed=seed)
    pass

def _evaluate(agent, save_dir, evaluation_epsiode, seed=None, task_1="PongNoFrameskip-v4", task_2="BeamRiderNoFrameskip-v4"):
    assert "NoFrameskip" in task_1 or "NoFrameskip" in task_2, "The string does not contain 'NoFrameskip'"

    print(20*"=", "Evaluation started", 20*"=")
    for env_name_eval in [task_1, task_2]:
        evaluation_data = evaluate(model=agent.kb_model, env_name=env_name_eval, save_dir=save_dir, num_episodes=evaluation_epsiode, seed=seed)
        print(f"Steps: {steps[env_name_eval]}, Frames in {env_name_eval}: {frame_number_eval[env_name_eval]}, Evaluation_score: {evaluation_data}")
        wandb.log({f"Evaluation_score_{env_name_eval}":evaluation_data})
    print(20*"=","Evaluation completed", 20*"=")
    pass

def environment_wrapper(env_name, mode="rgb_array", clip_rewards=True):
    
    if "StarGunner" in env_name:
        # "Pong" already has the desired action space
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3,  # LEFT
        }
    if "BeamRider" in env_name:
        # Adjusted action maps for "BeamRider" 
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 3,  # RIGHT
            3: 4,  # LEFT
        }
    if "SpaceInvaders" in env_name:
        # Adjusted action maps for "SpaceInvaders"
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3   # LEFT
        }

    env = UnifiedActionWrapper(gym.make(env_name, render_mode=mode), action_map=action_map)
    # env = gym.make(env_name, render_mode=mode, full_action_space=True)
    env = AtariPreprocessing(env=env, scale_obs=True)
    env = FrameStack(env=env, num_stack=4)
    env = EpisodicLifeEnv(env=env)
    if clip_rewards:
        env = ClipRewardEnv(env=env)
    return env

class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr