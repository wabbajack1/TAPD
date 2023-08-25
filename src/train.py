#!/Users/kerekmen/miniconda3/envs/agnostic_rl/bin/python
import wandb
import torch
import numpy as np
from agent import Agent
import datetime
from pathlib import Path
import torch.multiprocessing as mp
import random
from commons.EWC import EWC
import os
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor, as_completed
import arguments
from utils import evaluate, test_disitillation


print(f"Torch version: {torch.__version__}")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB_DIR"] = '..' # write path of wandb i.e. from working dir
os.environ['WANDB_MODE'] = 'online'

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
        project="Atari",
        entity="agnostic",
        config=args,
        notes=args.notes,
        # mode="disabled",
        # id="nd07r8xn",
        # resume="allow"
    )
    
    # Environments for multitask
    #environments = ["PongNoFrameskip-v4", 'StarGunnerNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    #environments = ['SpaceInvadersNoFrameskip-v4', "PongNoFrameskip-v4", 'StarGunnerNoFrameskip-v4']
    environments = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
    # environments = ["PongNoFrameskip-v4"]
    
    # args
    seed = wandb.config["seed"]
    set_seed(seed)
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
    ewclambda = wandb.config["ewclambda"]
    load_step_active = wandb.config["load_step_active"]
    load_step_kb = wandb.config["load_step_kb"]
    load_path = wandb.config["load_path"]
    mode = wandb.config["mode"]
    visits = wandb.config["visits"]
    ewc_start = wandb.config["ewc_start_timestep_after"]
    use_gpu = wandb.config["gpu"]

    # create path for storing meta data of the agent (hyperparams, video)
    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # create agent
    agent = Agent(use_gpu, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, seed, resume=False)
    
    ###################### start progress and compress algo #########################
    progress_and_compress(visits=visits, agent=agent, environments=environments, max_steps_progress=max_steps_progress, max_steps_compress=max_steps_compress, save_dir=save_dir, evaluation_steps=evaluate_nmb, batch_size_fisher=batch_size_fisher, batch_number_fisher=batch_number_fisher, ewcgamma=ewcgamma, ewclambda=ewclambda, ewc_start=ewc_start, seed=seed)

def progress_and_compress(visits, agent, environments, max_steps_progress, max_steps_compress, save_dir, evaluation_steps, batch_size_fisher, batch_number_fisher, ewcgamma, ewclambda, ewc_start, seed):
    visits = visits # visit each task x times
    for visit in range(visits):
        
        for env_name in environments:
        
            # Reinitialize workers for the new environment
            print("############## Creating workers ##############")
            agent.reinitialize_workers(env_name)
            
            print(f"############## Training on {env_name} ##############")
            # ############## progress activity ##############        
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
            
            ############## calculate ewc-online to include for next compress activity ##############
            # After learning each task, update EWC
            # latest_env = environment_wrapper(save_dir=save_dir, env_name=env_name, video_record=False)
            # init the first computation of the fisher, i.e. because later we compute only the running average
            # compute the fim based on the current policy, because otherwise the fim would be not a good estimate (becaue on-policy i.e. without replay buffer)
            # collect training data from current kb knowledge policy
            print("---- Collect rollout for EWC ----")
            for k in range(batch_number_fisher):
                # with ThreadPoolExecutor(max_workers=len(agent.workers)) as executor:
                # Submit tasks to the executor and collect results based on the current policy of kb knowledge
                # futures = [executor.submit(agent.collect_batch, worker, "Compress", batch_size_fisher) for worker in agent.workers]
                # batches = [f.result() for f in as_completed(futures)]

                for worker in agent.workers:
                    states, actions, true_values = agent.collect_batch(worker, "Compress", batch_size_fisher, len(agent.workers))
                    for i, _ in enumerate(states):
                        agent.memory.push(
                            states[i],
                            actions[i],
                            true_values[i]
                        )
                
            if agent.ewc_init:
                # take the latest env and calculate the fisher
                ewc = EWC(agent=agent, model=agent.kb_model, ewc_lambda=ewclambda, ewc_gamma=ewcgamma, device=agent.device, env_name=env_name, ewc_start_timestep_after=ewc_start)
                agent.ewc_init = False
            else: # else running calulaction taken the last fisher into consideration
                ewc.update(agent, agent.kb_model, env_name) # update the fisher after learning the current task. The current task becomes in the next iteration the previous task
                
            #agent.memory.delete_memory() # delete the data which was created for the current iteration from the workers
            agent.active_model.lateral_connections = True
            agent.resume = False # leaf this line here!
            
            # eval kb after learning/training     
            print(20*"=", "Evaluation started", 20*"=")
            for env_name_eval in environments:
                evaluate(model=agent.kb_model, env_name=env_name_eval,  num_steps=evaluation_steps, seed=seed)
            print(20*"=","Evaluation completed", 20*"=")

            # reset weights after each task
            agent.active_model.reset_weights(seed=seed)
            print(f"############## VISIT - {visit} ################ END OF TASK - {env_name}##############################\n")
        
    print("Training completed.\n")        

if __name__ == "__main__":
    args = arguments.get_args()
    main(args)

