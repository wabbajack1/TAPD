from commons.model import KB_Module, Active_Module
from commons.worker import Worker
from commons.memory.memory import Memory
import torch
import wandb
from commons.EWC import EWC
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
        self.active_model = Active_Module(self.device, env).to(self.device)
        self.kb_model = KB_Module(self.device, env).to(self.device)
        
        self.active_optimizer = torch.optim.RMSprop(self.active_model.parameters(), lr=self.lr, eps=1e-5)
        self.kb_optimizer = torch.optim.RMSprop(self.kb_model.parameters(), lr=self.lr, eps=1e-5)

    def create_workers(self, env_name):
        """ create workers for the env

        Args:
            env_name (string): env name

        Returns:
            list: list of workers 
        """
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.active_model, self.batch_size, self.gamma))
        
        return self.workers

    def reinitialize_workers(self, env_name):
        """Reinitialize the workers for a new environment."""
        self.ewc_flag = True
        self.workers = []
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.active_model, self.batch_size, self.gamma))

    def reflect(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        states, actions, true_values = self.memory.pop_all()
        values, log_probs, entropy = self.active_model.evaluate_action(states, actions)

        values = torch.squeeze(values)
        log_probs = torch.squeeze(log_probs)
        entropy = torch.squeeze(entropy)
        true_values = torch.squeeze(true_values)

        advantages =  true_values - values        
        critic_loss = advantages.pow(2).mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        total_loss = (self.critic_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)

        self.active_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 0.5)
        self.active_optimizer.step()
        
        return values.mean().item()

    def compress(self, env):
        criterion = torch.nn.KLDivLoss()
        ewc_lambda = 1000

        if self.ewc_flag:
            ewc_loss = EWC(env, self.kb_model).penalty(ewc_lambda, self.kb_model)
        else:
            ewc_loss = 0

        states, actions, true_values = self.memory.pop_all()
        _, log_probs_active, _ = self.active_model.evaluate_action(states, actions)
        _, log_probs_kb, _ = self.kb_model.evaluate_action(states, actions)

        kl_loss = criterion(torch.log(log_probs_kb), log_probs_active.detach()) + ewc_loss

        self.active_optimizer.zero_grad()
        self.kb_optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.kb_optimizer.parameters(), 0.5)
        self.kb_optimizer.step()
        
        return kl_loss.item()

    def progress_training(self, max_frames):
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
                
            value = self.reflect()  # Changed 'reflect(memory)' to 'self.reflect()'
            wandb.log({"Critic value": value}, commit=False)

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
