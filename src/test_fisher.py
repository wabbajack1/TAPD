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
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ["WANDB_DIR"] = '..' # write path of wandb i.e. from working dir
os.environ['WANDB_MODE'] = 'online'
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
        project="atari_single_task",
        entity="agnostic",
        config=args,
        notes=args.notes,
        mode="disabled",
        #id="nd07r8xn",
        #resume="allow"
    )
    
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

    # create path for storing meta data of the agent (hyperparams, video)
    save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    # create agent
    agent = Agent(True, learning_rate, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, seed, resume=False)
    
    ###################### start progress and compress algo #########################
    if load_path is not None:
        agent.load_active(load_path, load_step_active, mode)
        agent.load_kb(load_path, load_step_kb, mode)
   
    #progress_and_compress(visits=visits, agent=agent, environments=environments, max_steps_progress=max_steps_progress, max_steps_compress=max_steps_compress, save_dir=save_dir, evaluation_epsiode=evaluate_nmb, batch_size_fisher=batch_size_fisher, batch_number_fisher=batch_number_fisher, ewcgamma=ewcgamma, ewclambda=ewclambda, seed=seed)
    test_disitillation(agent=agent, max_steps_compress=max_steps_compress, save_dir=save_dir, evaluation_epsiode=evaluate_nmb, batch_size_fisher=batch_size_fisher, batch_number_fisher=batch_number_fisher, ewcgamma=ewcgamma, ewclambda=ewclambda, ewc_start=ewc_start, seed=seed, eval_after_compress=True)      

if __name__ == "__main__":
    args = arguments.get_args()
    main(args)

