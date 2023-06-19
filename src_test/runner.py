import sys
sys.path.append("../venv/lib/python3.10/site-packages/")
import agent
import gym
import wandb
import argparse
import torch
import numpy as np
from agent import Agent
import common.wrappers
import datetime
from pathlib import Path
import torch.multiprocessing as mp
from commons.model import Active_Module
import random
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from commons.EWC import EWC
import copy 
import os
from commons.model import weight_reset, init_weights
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

print(f"Torch version: {torch.__version__}")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["WANDB_DIR"] = '..' # write path of wandb i.e. from working dir
frame_number_eval = {}
steps = {}


def set_seed(seed: int = 44) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}\n")

def main(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="run_pandc_atari",
        entity="agnostic",
        config=args,
        mode="disabled",
        #id="nd07r8xn",
        #resume="allow"
    )
    seed = wandb.config["seed"]
    set_seed(seed)
    environments = ["PongNoFrameskip-v4", 'StarGunnerNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    max_steps_progress = wandb.config["max_steps_progress"]
    max_steps_compress = wandb.config["max_steps_compress"]
    batch_size = wandb.config["batch_size"]
    learning_rate = wandb.config["learning_rate"]
    gamma = wandb.config["gamma"]
    entropy_coef = wandb.config["entropy_coef"]
    critic_coef = wandb.config["critic_coef"]
    no_of_workers = wandb.config["workers"]
    eps = wandb.config["epsilon"]
    evaluate_nmb = wandb.config["evaluate"]
    batch_size_fisher = wandb.config["batch_size_fisher"]
    batch_number_fisher = wandb.config["batch_number_fisher"]
    ewcgamma = wandb.config["ewcgamma"]

    # create path for storing meta data of the agent (hyperparams, video)
    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # create agent
    agent = Agent(True, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, seed, resume=False)

    ###################### start progress and compress algo #########################
    # load agent if required crash and continue training
    # if agent.load_active(100_000) and agent.resume:
    #     print("Load successful!")
    # else:
    #     print("Load unsuccessful!")
   
    progress_and_compress(agent=agent, environments=environments, max_steps_progress=max_steps_progress, max_steps_compress=max_steps_compress, save_dir=save_dir, evaluation_epsiode=evaluate_nmb, batch_size_fisher=batch_size_fisher, batch_number_fisher=batch_number_fisher, ewcgamma=ewcgamma, seed=seed)      

def environment_wrapper(save_dir, env_name, video_record=False, clip_rewards=True):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """

    env = common.wrappers.make_atari(env_name, full_action_space=True)
    env = common.wrappers.wrap_deepmind(env, scale=True, clip_rewards=clip_rewards) 
    if video_record:
        path = (save_dir / "video" / "vid.mp4")
        env = gym.wrappers.Monitor(env, path, force=True, video_callable=lambda episode_id: True)
    env = common.wrappers.wrap_pytorch(env)
    return env

def progress_and_compress(agent, environments, max_steps_progress, max_steps_compress, save_dir, evaluation_epsiode, batch_size_fisher, batch_number_fisher, ewcgamma, seed):
    visits = 3 # visit each task x times
    for visit in range(visits):

        for env_name in environments:        
        
            # Reinitialize workers for the new environment
            print("############## Creating workers ##############")
            agent.reinitialize_workers(env_name)
            
            print(f"############## Training on {env_name} ##############")
            stop_value = env_name # current env name (in continual learning setup its a task)
            
            ############## progress activity ##############        
            if agent.resume != True: # check if run chrashed if yes resume the state of training before crash
                print(f"############## Progress phase - to step: {max_steps_progress}")
                agent.progress_training(max_steps_progress)
            
            print(f"############## NEXT PHASE ##############")

            ############## compress activity ##############
            print(f"############## Compress phase - to step: {max_steps_compress}")
            if agent.ewc_init == True:
                print("Distillation")
                agent.compress_training(max_steps_compress, None)
            else:
                print("Distillation + EWC")
                agent.compress_training(max_steps_compress, ewc)
            
            # eval kb after learning/training     
            print(20*"=", "Evaluation started", 20*"=")
            for env_name_eval in environments:
                evaluation_data = evaluate(model=agent.kb_model, env_name=env_name_eval, save_dir=save_dir, num_episodes=evaluation_epsiode, seed=seed)
                avg_score = evaluation_data
                
                print(f"Steps: {steps[env_name_eval]}, Frames in {env_name_eval}: {frame_number_eval[env_name_eval]}, Evaluation score: {avg_score}")
                wandb.log({f"Evaluation score-{env_name_eval}-{agent.kb_model.__class__.__name__}": avg_score, "Frame-# Evaluation": frame_number_eval[env_name_eval], "Steps-# Evaluation":steps[env_name_eval]})      
            print(20*"=","Evaluation completed", 20*"=")
            
            ############## calculate ewc-online to include for next compress activity ##############
            # After learning each task, update EWC
            # latest_env = environment_wrapper(save_dir=save_dir, env_name=env_name, video_record=False)
            # init the first computation of the fisher, i.e. because later we compute only the running average
            # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (becaue on-policy i.e. without replay buffer)
            # collect training data from current kb knowledge policy
            print("---- Collect rollout for EWC ----")
            for k in range(batch_number_fisher):
                with ThreadPoolExecutor(max_workers=len(agent.workers)) as executor:
                    # Submit tasks to the executor and collect results based on the current policy of kb knowledge
                    futures = [executor.submit(agent.collect_batch, worker, "Compress", batch_size_fisher) for worker in agent.workers]
                    batches = [f.result() for f in as_completed(futures)]
                    
                    
                for j, (states, actions, true_values) in enumerate(batches):
                    for i, _ in enumerate(states):
                        agent.memory.push(
                            states[i],
                            actions[i],
                            true_values[i]
                        )
                #print("----->", len(batches[0][0]), k)
                
            if agent.ewc_init:
                # take the latest env and calculate the fisher
                ewc = EWC(agent=agent, model=agent.kb_model, ewc_gamma=ewcgamma, device=agent.device, env_name=env_name)
                agent.ewc_init = False
            else: # else running calulaction taken the last fisher into consideration
                ewc.update(agent, agent.kb_model, env_name) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
                
            # reset weights after each task
            agent.active_model.reset_weights()
            #agent.memory.delete_memory() # delete the data which was created for the current iteration from the workers
            agent.active_model.lateral_connections = True
            agent.resume = False          
            print(f"############## VISIT - {visit} ################ END OF TASK - {env_name}##############################\n")
        
    print("Training completed.\n")        

def evaluate(model, env_name, save_dir, num_episodes, seed):
    
    # collect frame across all visits
    global frame_number_eval
    global steps
    
    # init env but do not clip the rewards in eval phase                    
    env = environment_wrapper(save_dir, env_name=env_name, video_record=False, clip_rewards=False)
    
    evaluation_scores = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_reward_orignal = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model.act(state_tensor)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if env_name not in steps:
                steps[env_name] = 0
            else:
                steps[env_name] += 1
                
            if done:
                if env_name not in frame_number_eval:
                    frame_number_eval[env_name] = 0
                else:
                    frame_number_eval[env_name] += info["episode_frame_number"]
            
        evaluation_scores.append(episode_reward)

    return np.mean(evaluation_scores) #, np.mean(evaluation_scores_original)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5,
        help="Batch size"
    )
    
    parser.add_argument(
        "-mfp",
        "--max_steps_progress",
        type=int,
        default=1000_000,
        help="Number of frames for progress phase")
    
    parser.add_argument(
        "-mfc",
        "--max_steps_compress",
        type=int,
        default=1000_000,
        help="Number of frames for compress phase (expected to be smaller number than mfp)")

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0007,
        help="Learning rate")

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.99,
        help="gamma discount value")
    
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=0.00001,
        help="epsilin decay for rms optimizer")

    parser.add_argument(
        "-ent",
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy coef value for exploartion")

    parser.add_argument(
        "-cri",
        "--critic_coef",
        type=float,
        default=0.5,
        help="critic coef value")

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="number of workers for running env")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=44,
        help="seed number")
    
    parser.add_argument(
        "-eval",
        "--evaluate",
        type=int,
        default=10,
        help="Run test with #-of episodes; The episodes get avg over the # of episodes provided")
    
    parser.add_argument(
        "-bF",
        "--batch_size_fisher",
        type=int,
        default=32,
        help="Batch size for calculating the estimate of the fisher")
    
    parser.add_argument(
        "-bFnumber",
        "--batch_number_fisher",
        type=int,
        default=100,
        help="Numbers of batches for calculating the estimate of the fisher")
    
    parser.add_argument(
        "-ewcgamma",
        "--ewcgamma",
        type=float,
        default=0.99,
        help="This is the decaying factor of the online-ewc algorithm, where ewcgamma = 1 indicates older tasks are more important than newer one")
    
    args = parser.parse_args()
    #mp.set_start_method('forkserver')
    main(args)

