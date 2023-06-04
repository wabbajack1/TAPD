#import sys
#sys.path.append("/Users/KerimErekmen/Desktop/Praesentation/Studium/Bachelor/Thesis/agnostic_rl-main/venv/lib/python3.10/site-packages/")
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

class Agent:
    def __init__(self, use_cuda, lr, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, eps, save_dir, resume):
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

    @staticmethod
    def collect_batch(worker, mode):
        return worker.get_batch(mode)
    
    def create_worker(self, i, env_name):
        worker = Worker(env_name, {"Progress": self.progNet, "Compress": self.kb_model}, self.batch_size, self.gamma, self.device, i)
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
        states, actions, true_values = self.memory.pop_all()
        self.memory.delete_memory()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        values_list = []
        
        for batch_states, batch_actions, batch_true_values in dataloader:
            batch_states = batch_states.to(self.device)
            batch_actions = batch_actions.to(self.device)
            batch_true_values = batch_true_values.to(self.device)
            #print(f"batch_states on {batch_states.shape}, batch_actions on {batch_actions.shape}, batch_true_values on {batch_true_values.shape}\n")
            
            values, log_probs, entropy = self.progNet.evaluate_action(batch_states, batch_actions) # inference of active column via kb column

            values = torch.squeeze(values)
            log_probs = torch.squeeze(log_probs)
            entropy = torch.squeeze(entropy)
            batch_true_values = torch.squeeze(batch_true_values)
            
            #print(values.shape, log_probs.shape, entropy.shape, batch_true_values.shape, entropy)
            
            advantages = batch_true_values - values
            critic_loss = advantages.pow(2).mean()

            actor_loss = -(log_probs * advantages.detach()).mean()
            total_loss = (self.critic_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)

            self.active_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 1)
            self.active_optimizer.step()

            values_list.extend(values.tolist())

        return np.mean(values_list).item()


    def compress(self, ewc):
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        ewc_lambda = 150
        total_kl_loss = 0

        states, actions, true_values = self.memory.pop_all() # take the same data again
        self.memory.delete_memory()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        values_list_kb = []
        values_list_ac = []
        for batch_states, batch_actions, _ in dataloader:
            batch_states = batch_states.to(self.device)
            batch_actions = batch_actions.to(self.device)            
            
            # calc infernce for loss calc
            values_active, log_probs_active, _ = self.progNet.evaluate_action(batch_states, batch_actions) # kb column is the last one with operation in the prognet
            values_kb, log_probs_kb, _ = self.kb_model.evaluate_action(batch_states, batch_actions)
            
            kl_loss = criterion(log_probs_kb, log_probs_active.detach())
            
            # calc ewc loss after every update and protected the weights w.r.t. the previous task 
            if ewc is not None:
                self.ewc_loss = ewc.penalty(ewc_lambda, self.kb_model) # the second argument needs the paramters of the model which is protected. The second argument suggests paramaters space during training, because in the ewc algo the old paramter was saved as a reference point in the paramters space for not converging to far from it
                #print("ewc loss", self.ewc_loss)
                total_loss = kl_loss + self.ewc_loss
            else:
                total_loss = kl_loss
                #print(kl_loss)
                
            
            #print("----->", torch.log(log_probs_kb).shape, log_probs_active.detach().shape)

            # calulate the gradients
            self.active_optimizer.zero_grad()
            self.kb_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kb_model.parameters(), 1)
            
            # make only step in kb column
            self.kb_optimizer.step()

            total_kl_loss += total_loss.item()
            values_list_kb.extend(values_kb.tolist())
            values_list_ac.extend(values_active.tolist())
            
        return np.mean(values_list_kb).item(), np.mean(values_list_ac).item(), (total_kl_loss / len(dataloader))


    def progress_training(self, max_frames, offset):
        frame_idx = 0
        data_value = []
        data_rewards = []
        updates = 0
        last_saved_frame_idx = 0
        self.active_model.train()
        self.kb_model.freeze_parameters()
        self.active_model.unfreeze_parameters()  

        while frame_idx < max_frames:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                # Submit tasks to the executor and collect results
                futures = [executor.submit(self.collect_batch, worker, "Progress") for worker in self.workers]
                batches = [f.result() for f in as_completed(futures)]

            for j, (states, actions, true_values, _) in enumerate(batches):
                for i, _ in enumerate(states):
                    self.memory.push(
                        states[i],
                        actions[i],
                        true_values[i]
                    )

            frame_idx += self.batch_size * len(self.workers)
            value = self.progress()
            data_value.append(value)
            wandb.log({"Critic value": np.mean(data_value[-100:])})
            updates += 1
            
            # save active model weights and optimizer status every 100_000 frames
            if (frame_idx - last_saved_frame_idx) >= 100_000:
                print(f"Save Active in Frame-#: {frame_idx+offset}\n")
                self.save_active(frame_idx+offset)
                last_saved_frame_idx = frame_idx

    def compress_training(self, max_frames, ewc, offset):
        frame_idx = 0
        last_saved_frame_idx = 0
        data_value_kb = []
        data_value_ac = []
        self.kb_model.train()  
        self.active_model.freeze_parameters()
        self.kb_model.unfreeze_parameters()  
        
        while frame_idx < max_frames:
            
            # collect training data
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                # Submit tasks to the executor and collect results
                futures = [executor.submit(self.collect_batch, worker, "Progress") for worker in self.workers]
                batches = [f.result() for f in as_completed(futures)]
                
                for j, (states, actions, true_values, _) in enumerate(batches):
                    for i, _ in enumerate(states):
                        self.memory.push(
                            states[i],
                            actions[i],
                            true_values[i]
                        )
            
            frame_idx += self.batch_size * len(self.workers)
            value_kb, value_ac, loss = self.compress(ewc)
            data_value_kb.append(value_kb)
            data_value_ac.append(value_ac)
            #print(f"Distillation loss: {loss}")
            wandb.log({"Distillation loss": loss, "Compress Value KB": np.mean(data_value_kb[-100:]), "Compress Value AC": np.mean(data_value_ac[-100:])})
            
            
            # # Only for: collect rewards from kb and log it (not training data!)
            # with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            #     # Submit tasks to the executor and collect results
            #     futures = [executor.submit(self.collect_batch, worker, "Compress") for worker in self.workers]
            #     _ = [f.result() for f in as_completed(futures)]
                
            # save active model weights and optimizer status every 100_000 frames
            if (frame_idx - last_saved_frame_idx) >= 100_000:
                print(f"Save KB in Frame-#: {frame_idx+offset}\n")
                self.save_kb(frame_idx+offset)
                last_saved_frame_idx = frame_idx

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

    def load_active(self, step):
        load_path = Path(f"../checkpoints/2023-05-23T20-03-49/active_model_{int(step)}.chkpt")

        if load_path.exists():
            checkpoint = torch.load(load_path)
            self.active_model.load_state_dict(checkpoint["model"])
            self.active_optimizer.load_state_dict(checkpoint["optimizer"])
            
            print(f"Active net loaded from {load_path}")
            return True  # Indicate successful load
        else:
            print(f"No active net found at {load_path}")
            return False  # Indicate unsuccessful load
        
    def load_kb(self, step):
        load_path = Path(f"../checkpoints/2023-05-23T20-03-49/kb_model_{int(step)}.chkpt")

        if load_path.exists():
            checkpoint = torch.load(load_path)
            self.kb_model.load_state_dict(checkpoint["model"])
            self.kb_optimizer.load_state_dict(checkpoint["optimizer"])
            
            print(f"Active net loaded from {load_path}")
            return True  # Indicate successful load
        else:
            print(f"No active net found at {load_path}")
            return False  # Indicate unsuccessful load