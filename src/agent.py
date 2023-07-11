from commons.model import KB_Module, Active_Module, ProgressiveNet
from commons.worker import Worker
from commons.memory.memory import Memory
import torch
import wandb
from commons.EWC import EWC
from commons.memory.CustomDataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import deque

class Agent:
    def __init__(self, use_cuda, lr, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, seed, resume):
        """In this implementation, we initialize the agent, which comprises various models, such as artificial neural networks (ANNs). 
        The agent features an active column, responsible for utilizing the model to accomplish specific tasks, and a knowledge base 
        column, representing the agent's memory. Through the forward model of the Intrinsic Curiosity Module (ICM), the 
        agent provides itself with intrinsic rewards.
        
        It is important to note that the agent maintains a consistent number of output neurons for each task, corresponding 
        to the same number of control units, even if a particular task does not require all of the available units.

        Args:
            use_cuda (_type_): _description_
            lr (_type_): _description_
            gamma (_type_): _description_
            entropy_coef (_type_): _description_
            critic_coef (_type_): _description_
            no_of_workers (_type_): _description_
            batch_size (_type_): _description_
            eps (_type_): _description_
            save_dir (_type_): _description_
        """
        self.seed = seed
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.no_of_workers = no_of_workers
        self.workers = []
        self.batch_size = batch_size
        self.ewc_init = True # is set to false after the first distillation of a task
        self.save_dir = save_dir
        self.wandb = wandb
        self.ewc_loss = 0
        self.resume = resume # continue with training state before crash

        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Cuda available: {torch.cuda.is_available()}, Set to: {use_cuda}")
        if self.device.type == 'cuda':
            print(f"\n=========== Run model on cuda ===========\n")

        # init active net, kb net, prog net and memory
        self.memory = Memory()
        self.active_model = Active_Module(self.device, lateral_connections=False).to(self.device)
        self.kb_model = KB_Module(self.device).to(self.device)
        self.progNet = ProgressiveNet(self.kb_model, self.active_model).to(self.device)

        # seperate optimizers because freezing method quite does not work, thererfore update only required nets
        self.kb_optimizer = torch.optim.RMSprop(self.kb_model.parameters(), lr=self.lr, eps=eps)
        self.active_optimizer = torch.optim.RMSprop(self.active_model.parameters(), lr=self.lr, eps=eps)
        self.progNet_optimizer = torch.optim.RMSprop(self.progNet.parameters(), lr=self.lr, eps=eps)

    @staticmethod
    def collect_batch(worker, mode, batch_size):
        return worker.get_batch(mode, batch_size)
    
    def create_worker(self, i, env_name):
        worker = Worker(env_name, {"Progress": self.progNet, "Compress": self.kb_model}, self.batch_size, self.gamma, self.device, self.seed, i)
        print(f"Worker {i} created\n")
        return worker

    def create_worker_parallel(self, env_name):
        """ create workers for the env

        Args:
            env_name (string): env name

        Returns:
            list: list of workers 
        """
        with ThreadPoolExecutor() as executor:
            # Create a list of futures with the worker creation tasks
            futures = [executor.submit(self.create_worker, i, env_name) for i in range(self.no_of_workers)]

            # Wait for all the tasks to complete and get the results
            self.workers = [future.result() for future in as_completed(futures)]

        return self.workers

    def reinitialize_workers(self, env_name):
        """Reinitialize the workers for a new environment."""
        self.workers = []
        self.create_worker_parallel(env_name=env_name)

    def progress(self):
        # fetch experience
        states, actions, true_values = self.memory.pop_all()
        self.memory.delete_memory()
        
        #### begin of calculation ####
        states = states.to(self.device)
        actions = actions.to(self.device)
        true_values = true_values.to(self.device)
        
        values, log_probs, entropy = self.progNet.evaluate_action(states, actions) # inference of active column via kb column
        values = torch.squeeze(values)
        log_probs = torch.squeeze(log_probs)
        entropy = torch.squeeze(entropy)
        true_values = torch.squeeze(true_values)
        
        # A(s_t, a_t) = r_t+1 + gamma*V(s_t+1, phi) - V(s_t, phi)
        advantages = true_values - values
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        critic_loss = advantages.pow(2).mean()
        
        # mean(log(pi(a_t, s_t))* A(s_t, a_t))
        actor_loss = -(log_probs * advantages.detach()).mean()
        total_loss = (self.critic_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)
        
        self.active_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 0.5)
        self.active_optimizer.step()
        
        # print(advantages_normlized, "\n", advantages)
        # print(f"loss {total_loss}, actor_loss {actor_loss}, critic_loss {critic_loss}, entropy {entropy}")
        # print(f"states on {states.shape}, actions on {actions.shape}, true_values on {true_values.shape}")
        return values.mean().item(), critic_loss.item(), actor_loss.item(), entropy.item()

    def compress(self, ewc):
        # specificy loss function
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        
        # fetch experience
        states, actions, true_values = self.memory.pop_all() # take the same data again
        self.memory.delete_memory()
        
        #### begin of calculation ####
        states = states.to(self.device)
        actions = actions.to(self.device)            
        
        # imitation of kb to active network
        _, log_probs_active, _ = self.progNet.evaluate_action(states, actions) # kb column is the last one with operation in the prognet
        _, log_probs_kb, _ = self.kb_model.evaluate_action(states, actions)
        
        # calculate the loss function
        kl_loss = criterion(log_probs_kb.unsqueeze(0), log_probs_active.unsqueeze(0).detach())
        
        # calc ewc loss after every update and protected the weights w.r.t. the previous task
        if ewc is not None:
            # The second argument, crucial for EWC, guides parameter space during training using old parameters as reference to prevent excessive divergence
            self.ewc_loss = ewc.penalty(self.kb_model)
            total_loss = kl_loss + self.ewc_loss
        else:
            total_loss = kl_loss

        # calulate the gradients
        self.active_optimizer.zero_grad()
        self.kb_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.kb_model.parameters(), 0.5)
        self.kb_optimizer.step()
        
        print(kl_loss, self.ewc_loss)
        
        return total_loss.item(), kl_loss.item(), float(self.ewc_loss)

    def progress_training(self, max_steps):
        # watch variables
        steps_idx = 0
        updates = 0
        last_saved_steps_idx = 0
        
        # Initialize deque with max length 100 for logging data
        value_log = deque(maxlen=100)
        critic_loss_log = deque(maxlen=100)
        actor_loss_log = deque(maxlen=100)
        entropy_log = deque(maxlen=100)
        
        # freeze and train model
        self.active_model.train()
        self.kb_model.freeze_parameters()
        self.active_model.unfreeze_parameters()
        
        # iterate over the specifiec steps in the environment
        while steps_idx < max_steps:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                # Submit tasks to the executor and collect results
                futures = [executor.submit(self.collect_batch, worker, "Progress", None) for worker in self.workers]
                batches = [f.result() for f in as_completed(futures)] # nmb of workers (each worker has states, actions, true_values, ...)
            
            
            # iterate over each workers batch and push to memory (batches = (#-workers, states, actions, ...))
            for j, (states, actions, true_values) in enumerate(batches):
                for i, _ in enumerate(states):
                    self.memory.push(
                        states[i],
                        actions[i],
                        true_values[i]
                    )
                    
            steps_idx += self.batch_size * len(self.workers)
            value, critic_loss, actor_loss, entropy = self.progress() # train active column
            updates += 1
            
            # log the incoming data
            value_log.append(value)
            critic_loss_log.append(critic_loss)
            actor_loss_log.append(actor_loss)
            entropy_log.append(entropy)
            
            # save active model weights and optimizer status every 100_000 steps
            if (steps_idx - last_saved_steps_idx) >= 500_000:
                print(f"Save Active in step-# with updates = {updates}: {steps_idx}\n")
                self.save_active(steps_idx)
                last_saved_steps_idx = steps_idx
            
            if steps_idx % 10000 == 0: # log every 10000 steps
                value_mean = np.mean(value_log)
                critic_loss_mean = np.mean(critic_loss_log)
                actor_loss_mean = np.mean(actor_loss_log)
                entropy_mean = np.mean(entropy_log)
                
                wandb.log({
                    "Value": value_mean, 
                    "Critic Loss": critic_loss_mean, 
                    "Actor Loss": actor_loss_mean,
                    "Entropy": entropy_mean,
                    "Steps Progress": steps_idx
                })

    def compress_training(self, max_steps, ewc):
        # init values for run
        steps_idx = 0
        last_saved_steps_idx = 0
        updates = 0
        
        # Initialize deque with max length 100 for logging data
        total_loss_log = deque(maxlen=100)
        kl_loss_log = deque(maxlen=100)
        ewc_loss_log = deque(maxlen=100)
        
        # model training
        self.kb_model.train()  
        self.active_model.freeze_parameters()
        self.kb_model.unfreeze_parameters()
        
        while steps_idx < max_steps:
            # collect training data
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                # Submit tasks to the executor and collect results
                futures = [executor.submit(self.collect_batch, worker, "Progress", None) for worker in self.workers]
                batches = [f.result() for f in as_completed(futures)]
                
                for j, (states, actions, true_values) in enumerate(batches):
                    for i, _ in enumerate(states):
                        self.memory.push(
                            states[i],
                            actions[i],
                            true_values[i]
                        )
            
            steps_idx += self.batch_size * len(self.workers)
            total_loss, kl_loss, ewc_loss = self.compress(ewc) # train
            updates += 1
            
            # log incoming data
            total_loss_log.append(total_loss)
            kl_loss_log.append(kl_loss)
            ewc_loss_log.append(ewc_loss)
            
            # save active model weights and optimizer status every 100_000 steps
            if (steps_idx - last_saved_steps_idx) >= 500_000:
                print(f"Save KB in step-# with updates = {updates}: {steps_idx}\n")
                self.save_kb(steps_idx)
                last_saved_steps_idx = steps_idx
            
            if steps_idx % 10000 == 0: # log every 10000 steps
                
                # calcualte mean
                total_loss_mean = np.mean(total_loss_log)
                kl_loss_mean = np.mean(kl_loss_log)
                ewc_loss_mean = np.mean(ewc_loss_log)
                
                wandb.log({"Distillation loss (KL Loss + EWC loss)": total_loss_mean,
                           "KL Loss": kl_loss_mean,
                           "EWC loss": ewc_loss_mean,
                           "Steps Compress": steps_idx})

    def save_active(self, step):
        """step + self.inc_active = the last digit adds the step size to the step size

        Args:
            step (_type_): _description_
        """
        save_path = (self.save_dir / f"active_model_{int(step)}.chkpt")
        torch.save(dict(model=self.active_model.state_dict(), optimizer=self.active_optimizer.state_dict(), **self.wandb.config), save_path)
        print(f"Active net saved to {save_path}")
        
    def save_kb(self, step):
        """step + self.inc_kb = the last digit adds the step size to the step size

        Args:
            step (_type_): _description_
        """
        save_path = (self.save_dir / f"kb_model_{int(step)}.chkpt")
        torch.save(dict(model=self.kb_model.state_dict(), optimizer=self.kb_optimizer.state_dict(), **self.wandb.config), save_path)
        print(f"KB net saved to {save_path}")

    def load_active(self, load_path, load_step, mode="cpu"):
        load_path = Path(f"{load_path}/active_model_{int(load_step)}.chkpt")
        modified_state_dict = {}
        
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=torch.device(mode))
            
            for key, value in checkpoint["model"].items():
                # Exclude or rename keys as needed
                if key == 'adaptor.critic_adaptor.weight':
                    modified_state_dict['adaptor.critic_adaptor.0.weight'] = value
                elif key == 'adaptor.critic_adaptor.bias':
                    modified_state_dict['adaptor.critic_adaptor.0.bias'] = value
                elif key == 'adaptor.actor_adaptor.weight':
                    modified_state_dict['adaptor.actor_adaptor.0.weight'] = value
                elif key == 'adaptor.actor_adaptor.bias':
                    modified_state_dict['adaptor.actor_adaptor.0.bias'] = value
                else:
                    modified_state_dict[key] = value
            
            
            self.active_model.load_state_dict(modified_state_dict)
            #self.active_optimizer.load_state_dict(checkpoint["optimizer"])
            
            print(f"Active net loaded from {load_path}")
            return True  # Indicate successful load
        else:
            print(f"No Active net found at {load_path}")
            return False  # Indicate unsuccessful load
        
    def load_kb(self, load_path, load_step, mode):
        load_path = Path(f"{load_path}/kb_model_{int(load_step)}.chkpt")
        modified_state_dict = {}
        
        if load_path.exists():
            checkpoint = torch.load(load_path, map_location=torch.device(mode))
        
            # Filter out unwanted keys
            state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith('adaptor')}
        
            self.kb_model.load_state_dict(state_dict)
            #self.kb_optimizer.load_state_dict(checkpoint["optimizer"])
            
            print(f"kb net loaded from {load_path}")
            return True  # Indicate successful load
        else:
            print(f"No kb net found at {load_path}")
            return False  # Indicate unsuccessful load