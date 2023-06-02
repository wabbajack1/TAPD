import sys
sys.path.append("../venv/lib/python3.9/site-packages/")
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
        #mode="disabled",
        #id="nd07r8xn",
        #resume="allow"
    )
    
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
    set_seed(wandb.config["seed"])

    # create path for storing meta data of the agent (hyperparams, video)
    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # create agent
    agent = Agent(True, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, resume=True)
    
    ############### RUN ONLY ACTIVE COLUMN AND ONE TASK (FOR TESTING) ###############
    '''agent.create_worker_parallel(environments[0])
    agent.progress_training(max_frames)'''
    #################################################################################

    ###################### start progress and compress algo #########################
    trained_agent = progress_and_compress(agent=agent, environments=environments, max_frames_progress=max_frames_progress, max_frames_compress=max_frames_compress, save_dir=save_dir, evaluation_interval=evaluate_nmb, seed=wandb.config["seed"])
    # initial_state_dict = copy.deepcopy(agent.progNet.model_b.state_dict())
    # print(evaluate(agent.progNet.model_b, "StarGunnerNoFrameskip-v4", agent.device, None, num_episodes=10))
    # #agent.progNet.model_b.reinit_parameters(wandb.config["seed"])
    # set_seed(44)
    # agent.progNet.model_b.load_state_dict(initial_state_dict)
    # print(evaluate(agent.progNet.model_b, "StarGunnerNoFrameskip-v4", agent.device, None, num_episodes=10))

    # agent.load_active("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/checkpoints/2023-06-01T14-48-21", 100_000)

    # for env_name_eval in environments:
    #     evaluation_score = evaluate(agent.progNet, env_name_eval, agent.device, save_dir=save_dir)
    #     print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score[1]}")
    #     wandb.log({f"Evaluation score;{env_name_eval};{agent.progNet.model_b.__class__.__name__}": evaluation_score[1]})
        

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
    
    # load agent if required crash and continue training
    if agent.load_active(100_000) and agent.resume:
        print("Load successful!")
    else:
        print("Load unsuccessful!")
    
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
                agent.progress_training(evaluation_interval)
                
                #evaluate the perforamance of the agent after the training
                for env_name_eval in environments:
                    evaluation_score = evaluate(agent.progNet, env_name_eval, agent.device, save_dir=save_dir)
                    print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score[1]}")
                    wandb.log({f"Evaluation score;{env_name_eval};{agent.progNet.model_b.__class__.__name__}": evaluation_score[1]})

                # for env_name_eval in environments:
                #     evaluation_score = evaluate(agent.kb_model, env_name_eval, agent.device, save_dir=save_dir)
                #     print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score[1]}")
                #     wandb.log({f"Evaluation score;{env_name_eval};{agent.kb_model.__class__.__name__}": evaluation_score[1]})

        
        
        ############## compress activity ##############
        print(f"############## Compress phase with EWC.task-{env_name}")
        task_list = [environment_wrapper(save_dir=save_dir, env_name=environments[i], video_record=False) for i in range(environments.index(stop_value))] # list for ewc calculation for considerering the previous tasks
       
        # init the first computation of the fisher, i.e. because later we compute only the running average
        if agent.ewc_init and len(task_list) > 0:
            # take the latest env and calculate the fisher
            ewc = EWC(task_list[-1], agent.kb_model, ewc_gamma=0.99, device=agent.device)
            agent.ewc_init = False
            
        for frame_idx in range(0, max_frames_compress, evaluation_interval):
            print(f"############## Compress phase - to Frame: {frame_idx + evaluation_interval}")
            
            if agent.ewc_init == True:
                print("Distillation")
                agent.compress_training(evaluation_interval, None)
            else:
                # go through all previosly env for ewc calculation
                print("Distillation + EWC")
                agent.compress_training(evaluation_interval, ewc)
                
            for env_name_eval in environments:
                evaluation_score = evaluate(agent.kb_model, env_name_eval, agent.device, save_dir=save_dir)
                print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score[1]}")
                wandb.log({f"Evaluation score;{env_name_eval};{agent.kb_model.__class__.__name__}": evaluation_score[1]})
        
        #agent.memory.delete_memory() # delete the data which was created for the current iteration from the workers
        # After learning each task, update EWC
        latest_env = environment_wrapper(save_dir=save_dir, env_name=env_name, video_record=False)
        agent.active_model.lateral_connections = True
        agent.resume = False                
        
        if agent.ewc_init == False:
            ewc.update(agent.kb_model, latest_env) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
        
        agent.active_model.reinit_parameters(0)
        print("\n ############################## \n")
    
    print("Training completed.\n")

    # Evaluate the agents performance on each environment again after training on all environments
    for env_name in environments:
        evaluation_score = evaluate(agent.progNet, env_name, agent.device, save_dir=save_dir)
        print(f"Evaluation score after training on {env_name}: {evaluation_score}")
        
    print("Evaluation completed.\n")

    return agent

def evaluate(model, env, device, save_dir, num_episodes=10):
    env = environment_wrapper(save_dir, env_name=env, video_record=False)
    #print(f"Evaluate: {env.spec.id} with model {model.__class__.__name__}")
    
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
            next_state, reward, done = env.step(action)
            episode_reward += reward[0]
            episode_reward_orignal += reward[1]
            state = next_state

        evaluation_scores.append(episode_reward)
        evaluation_scores_original.append(episode_reward_orignal)

    return np.mean(evaluation_scores), np.mean(evaluation_scores_original)

def train_PC(env_dict, max_frames, agent):
    "Train all 5 taks using PNN"
    
    for task_id, env_task in env.items():
        print(f"==================TRAIN TASK {task_id}======================\n")
        
        # switch grad for progress and compress phase
        actor.unfreezeColumn(net1_actor)
        critic.unfreezeColumn(net1_critic)
        
        #net.freezeColumn(kb_column)
        #net.network_reset(kb_column)

        # train the agent
        agent.progress_training(max_frames)
        
        # switch grad for compress phase
        """net.unfreezeColumn(kb_column)
        net.freezeColumn(active_column)      
        
        if task_id == 0:
            loss = train_compress_normal(actor_critic, active_column, kb_column, task_id, device, env_task, optimizer, episodes, max_steps=max_steps, mem_experience_buffer=mem_experience_buffer, log_training=logs)
            loss_kb[task_id].append(loss)
            #test(net, active_column, device, test_loader[task_id])
        else:
            old_obs = old_obs + mem_experience_buffer_prev
            old_obs = random.sample(old_obs, k=round(len(mem_experience_buffer)/2))
            loss_ewc = train_compress_ewc(net, old_obs, ewc_lambda, active_column, kb_column, task_id, device, env_task, optimizer, episodes, max_steps=max_steps, mem_experience_buffer=mem_experience_buffer)
            loss_kb[task_id].append(loss_ewc)
            
            for i in range(task_id+1):
                env_test = env_dict[i]
                score = 0
                state = stack_frames(None, env_test.reset()[0], True)
            while True:
                #env_test.render()
                action_pred_logit_kb, action, log_prob_ac, _, value_ac = act(net, kb_column, state)
                next_state, reward, done, _,_ = env_test.step(action)
                score += reward
                state = stack_frames(state, next_state, False)
                if done:
                    print("You Final score is:", score)
                    score_kb[i].append(score)
                    break 
            env_test.close()
                    
                    
        mem_experience_buffer_prev = mem_experience_buffer
        
        net.freezeColumn(kb_column)
        net.freezeColumn(active_column)"""

        """# Serializing json
        json_kb = json.dumps(convert(score_kb), indent=4)
        json_ac = json.dumps(convert(score_ac), indent=4)
        
        # Writing to sample.json
        with open(f"/content/drive/MyDrive/Github/agnostic_rl/agent/score_kb_{ewc_lambda}.json", "w") as outfile:
            outfile.write(f"{net}\n\n{json_kb}")

        with open(f"/content/drive/MyDrive/Github/agnostic_rl/agent/score_ac_{ewc_lambda}.json", "w") as outfile:
            outfile.write(f"{net}\n\n{json_ac}")"""

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

