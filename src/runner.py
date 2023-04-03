#!/usr/bin/env python

import agent
import gym
import wandb
import argparse
import torch
import numpy as np
from agent import Agent
import sys
sys.path.append("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/lib/python3.9/site-packages/")
import common.wrappers

def main(args):
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="test_run_pong",
        entity="agnostic",
        config=args
    )

    max_frames = wandb.config["max_frames"]
    batch_size = wandb.config["batch_size"]
    learning_rate = wandb.config["learning_rate"]
    gamma = wandb.config["gamma"]
    entropy_coef = wandb.config["entropy_coef"]
    critic_coef = wandb.config["critic_coef"]
    no_of_workers = wandb.config["workers"]


    environments = ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    env = environment_wrapper(environments[0])
    
    # init the agent for later use
    agent = Agent(True, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, env)

    agent.create_workers(environments[0])
    agent.progress_training(max_frames)

    # start progress and compress algo
    #trained_agent = progress_and_compress(agent, environments, max_frames, evaluation_interval)


def environment_wrapper(env_name):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """
    env = common.wrappers.make_atari(env_name)
    env = common.wrappers.wrap_deepmind(env, scale=True)
    env = common.wrappers.wrap_pytorch(env)

    return env

def progress_and_compress(agent, environments, max_frames, evaluation_interval):
    for env_name in environments:
        print(f"Training on {env_name}")
        stop_value = env_name # current env name

        # progress activity
        for frame_idx in range(0, max_frames, evaluation_interval):
            agent.progress_training(evaluation_interval)

            for env_name_eval in environments:
                evaluation_score = evaluate(agent.active_model, env_name_eval)
                print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score}")
                wandb.log({f"evaluation_score_{env_name_eval}": evaluation_score})
        
        # Compress the knowledge
        for frame_idx in range(0, max_frames, evaluation_interval):
            output_list = [environments[i] for i in range(environments.index(stop_value))]

            # go through all previosly env for ewc calculation
            for running_env in output_list:
                env = environment_wrapper(running_env)
                agent.compress_training(evaluation_interval, env)
            
            for env_name_eval in environments:
                evaluation_score = evaluate(agent.kb_model, env_name_eval)
                print(f"Frame: {frame_idx}, Evaluation score: {evaluation_score}")
                wandb.log({f"evaluation_score_{env_name_eval}": evaluation_score})

        # Reinitialize workers for the new environment
        agent.reinitialize_workers(env_name)

    print("Training completed.")

    # Evaluate the agent's performance on each environment again after training on all environments
    for env_name in environments:
        evaluation_score = evaluate(agent, env_name)
        print(f"Final evaluation score on {env_name}: {evaluation_score}")

    return agent

def evaluate(model, env, num_episodes=10):
    evaluation_scores = []
    env = environment_wrapper(env_name=env_name)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model.act(state_tensor)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        evaluation_scores.append(episode_reward)

    return np.mean(evaluation_scores)

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
        "-mf",
        "--max_frames",
        type=int,
        default=1000_000,
        help="Number of frames")

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
    
    args = parser.parse_args()
    main(args)
