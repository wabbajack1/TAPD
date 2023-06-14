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

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["WANDB_DIR"] = '..' # write path of wandb i.e. from working dir

print(f"Torch version: {torch.__version__}")

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
    
    set_seed(wandb.config["seed"])
    
    environments = ["PongNoFrameskip-v4", 'StarGunnerNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    max_frames_progress = wandb.config["max_frames_progress"]
    max_frames_compress = wandb.config["max_frames_compress"]
    batch_size = wandb.config["batch_size"]
    learning_rate = wandb.config["learning_rate"]
    gamma = wandb.config["gamma"]
    entropy_coef = wandb.config["entropy_coef"]
    critic_coef = wandb.config["critic_coef"]
    no_of_workers = wandb.config["workers"]
    eps = wandb.config["epsilon"]
    evaluate_nmb = wandb.config["evaluate"]

    # create path for storing meta data of the agent (hyperparams, video)
    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # create agent
    agent = Agent(True, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, resume=False)

    ###################### start progress and compress algo #########################
    # load agent if required crash and continue training
    # if agent.load_active(100_000) and agent.resume:
    #     print("Load successful!")
    # else:
    #     print("Load unsuccessful!")
   
    progress_and_compress(agent=agent, environments=environments, max_frames_progress=max_frames_progress, max_frames_compress=max_frames_compress, save_dir=save_dir, evaluation_interval=evaluate_nmb, seed=wandb.config["seed"])
        

def environment_wrapper(save_dir, env_name, video_record=False):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """

    env = common.wrappers.make_atari(env_name, full_action_space=True)
    env = common.wrappers.wrap_deepmind(env, scale=True, clip_rewards=True) 
    if video_record:
        path = (save_dir / "video" / "vid.mp4")
        env = gym.wrappers.Monitor(env, path, force=True, video_callable=lambda episode_id: True)
    env = common.wrappers.wrap_pytorch(env)
    return env

def progress_and_compress(agent, environments, max_frames_progress, max_frames_compress, save_dir, evaluation_interval, seed):
    
    visit = 3
    for i in range(visit):

        for env_name in environments:        
        
            # Reinitialize workers for the new environment
            print("############## Creating workers ##############")
            agent.reinitialize_workers(env_name)
            
            print(f"############## Training on {env_name} ##############")
            stop_value = env_name # current env name (in continual learning setup its a task)
            
            ############## progress activity ##############        
            if agent.resume != True: # check if run chrashed if yes resume the state of training before crash
                for frame_idx in range(0, max_frames_progress, evaluation_interval):
                    print(f"############## Progress phase - to Frame: {frame_idx + evaluation_interval}")
                    agent.progress_training(evaluation_interval, offset=frame_idx)

            
            print(f"############## NEXT PHASE ##############")

            ############## compress activity ##############
            #task_list = [environment_wrapper(save_dir=save_dir, env_name=environments[i], video_record=False) for i in range(environments.index(stop_value))] # list for ewc calculation for considerering the previous tasks

            for frame_idx in range(0, max_frames_compress, evaluation_interval):
                print(f"############## Compress phase - to Frame: {frame_idx + evaluation_interval}")
                
                if agent.ewc_init == True:
                    print("Distillation")
                    agent.compress_training(evaluation_interval, None, offset=frame_idx)
                else:
                    # go through all previosly env for ewc calculation
                    print("Distillation + EWC")
                    agent.compress_training(evaluation_interval, ewc, offset=frame_idx)
                    
                # for env_name_eval in environments:
                #     evaluation_score = evaluate(agent.kb_model, env_name_eval, agent.device, save_dir=save_dir)
                #     print(f"Frame: {frame_idx + evaluation_interval}, Evaluation score: {evaluation_score}")
                #     wandb.log({f"Evaluation score;{env_name_eval};{agent.kb_model.__class__.__name__}": evaluation_score})
            
            #agent.memory.delete_memory() # delete the data which was created for the current iteration from the workers
            agent.active_model.lateral_connections = True
            agent.resume = False                
            
            # After learning each task, update EWC
            latest_env = environment_wrapper(save_dir=save_dir, env_name=env_name, video_record=False)
            
            # init the first computation of the fisher, i.e. because later we compute only the running average
            # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (online learn i.e. without replay buffer)
            if agent.ewc_init:
                # take the latest env and calculate the fisher
                ewc = EWC(task=latest_env, model=agent.kb_model, num_samples=1, ewc_gamma=0.65, device=agent.device)
                agent.ewc_init = False
            else: # else running calulaction
                ewc.update(agent.kb_model, latest_env) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
            
            # reset weights after each task
            agent.active_model.reset_weights()
            print(f"############## VISIT - {i} ################ END OF TASK - {env_name}##############################\n")
        
    print("Training completed.\n")

    # Evaluate the agents performance on each environment again after training on all environments
    print(20*"=", "END Evaluation started", 20*"=")
    for env_name in environments:
        evaluation_score = evaluate(agent.progNet, env_name, agent.device, save_dir=save_dir)
        print(f"Evaluation score after training on {env_name}: {evaluation_score}")
        
    print("Evaluation completed.\n")

def evaluate(model, env, device, save_dir, num_episodes=10):
    env = environment_wrapper(save_dir, env_name=env, video_record=False)
    
    evaluation_scores = []
    evaluation_scores_original = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_reward_orignal = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model.act(state_tensor.to(device))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            #episode_reward_orignal += reward[1]
            state = next_state

        evaluation_scores.append(episode_reward)
        #evaluation_scores_original.append(episode_reward_orignal)

    return np.mean(evaluation_scores)#, np.mean(evaluation_scores_original)

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
        "--max_frames_progress",
        type=int,
        default=1000_000,
        help="Number of frames for progress phase")
    
    parser.add_argument(
        "-mfc",
        "--max_frames_compress",
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
        default=100000,
        help="Run test after #-frames")
    
    args = parser.parse_args()
    #mp.set_start_method('forkserver')
    main(args)

