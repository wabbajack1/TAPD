from commons.model import KB_Module, Active_Module, ProgressiveNet
from commons.worker import Worker
from commons.memory.memory import Memory
import torch
import wandb
from commons.EWC import EWC
from commons.memory.CustomDataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

class Agent:
    def __init__(self, use_cuda, lr, gamma, entropy_coef, critic_coef, no_of_workers, batch_size, env):
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.no_of_workers = no_of_workers
        self.workers = []
        self.memory = Memory()
        self.batch_size = batch_size
        self.num_actions = env.action_space.n
        self.ewc_flag = False

        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            print(f"Run model on cuda\n")

        # init active and kb column
        self.active_model = Active_Module(self.device, env, lateral_connections=False).to(self.device)
        self.kb_model = KB_Module(self.device, env).to(self.device)
        self.progNet = ProgressiveNet(self.kb_model, self.active_model)

        # seperate optimizers because freezing method quit does not work, thererfore update only required nets
        self.active_optimizer = torch.optim.RMSprop(self.active_model.parameters(), lr=self.lr, eps=0.1)
        self.kb_optimizer = torch.optim.RMSprop(self.kb_model.parameters(), lr=self.lr, eps=0.1)


    def create_workers(self, env_name):
        """ create workers for the env

        Args:
            env_name (string): env name

        Returns:
            list: list of workers 
        """
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.progNet, self.batch_size, self.gamma))
        
        return self.workers

    def reinitialize_workers(self, env_name):
        """Reinitialize the workers for a new environment."""
        self.ewc_flag = True
        self.workers = []
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.progNet, self.batch_size, self.gamma))

    def progress(self):
        states, actions, true_values = self.memory.pop_all()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        values_list = []
        
        for batch_states, batch_actions, batch_true_values in dataloader:
            print(batch_states.shape, batch_actions.shape, batch_true_values.shape)
            values, log_probs, entropy = self.progNet.evaluate_action(batch_states, batch_actions) # inference of active column via kb column

            values = torch.squeeze(values)
            log_probs = torch.squeeze(log_probs)
            entropy = torch.squeeze(entropy)
            batch_true_values = torch.squeeze(batch_true_values)

            advantages = batch_true_values - values
            critic_loss = advantages.pow(2).mean()

            actor_loss = -(log_probs * advantages.detach()).mean()
            total_loss = (self.critic_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)

            self.active_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 1)
            self.active_optimizer.step()
            
            values_list.append(values)

        return values.mean().item()


    def compress(self, env):
        criterion = torch.nn.KLDivLoss()
        ewc_lambda = 1000

        states, actions, true_values = self.memory.pop_all()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_kl_loss = 0
        for batch_states, batch_actions, _ in dataloader:
            
            # calc ewc loss after every update
            if self.ewc_flag:
                ewc_loss = EWC(env, self.kb_model).penalty(ewc_lambda, self.kb_model)
            else:
                ewc_loss = 0
            
            # calc infernce for loss calc
            _, log_probs_active, _ = self.active_model.evaluate_action(batch_states, batch_actions)
            _, log_probs_kb, _ = self.kb_model.evaluate_action(batch_states, batch_actions)

            kl_loss = criterion(torch.log(log_probs_kb), log_probs_active.detach()) + ewc_loss

            self.active_optimizer.zero_grad()
            self.kb_optimizer.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kb_optimizer.parameters(), 1)
            self.kb_optimizer.step()

            total_kl_loss += kl_loss.item()

        return total_kl_loss / len(dataloader)


    def progress_training(self, max_frames):
        frame_idx = 0
        data = []
        while frame_idx < max_frames:
            for worker in self.workers:
                states, actions, true_values = worker.get_batch()
                for i, _ in enumerate(states):
                    self.memory.push(
                        states[i],
                        actions[i],
                        true_values[i]
                    )
                frame_idx += self.batch_size
                
            value = self.progress()  # Changed 'reflect(memory)' to 'self.reflect()'
            data.append(value)
            #wandb.log({"Critic value": np.mean(data[-100:])})

    def compress_training(self, max_frames, env):
        frame_idx = 0

        while frame_idx < max_frames:
            for worker in self.workers:
                states, actions, true_values = worker.get_batch()
                for i, _ in enumerate(states):
                    self.memory.push(
                        states[i],
                        actions[i],
                        true_values[i]
                    )
                frame_idx += self.batch_size
            
            self.compress(env)
