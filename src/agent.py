from commons.model import Model
from commons.worker import Worker
from commons.memory.memory import Memory
import torch

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

        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            print(f"Run model on cuda\n")

        self.model = Model(self.device, self.num_actions, env).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, eps=1e-5)

    def create_workers(self, env_name):
        """ create workers for the env

        Args:
            env_name (string): env name

        Returns:
            list: list of workers 
        """
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.model, self.batch_size, self.gamma))
        
        return self.workers

    def reinitialize_workers(self, env_name):
        """Reinitialize the workers for a new environment."""
        self.workers = []
        for _ in range(self.no_of_workers):
            self.workers.append(Worker(env_name, self.model, self.batch_size, self.gamma))

    def reflect(self):
        states, actions, true_values = self.memory.pop_all()
        values, log_probs, entropy = self.model.evaluate_action(states, actions)

        values = torch.squeeze(values)
        log_probs = torch.squeeze(log_probs)
        entropy = torch.squeeze(entropy)
        true_values = torch.squeeze(true_values)

        advantages =  true_values - values        
        critic_loss = advantages.pow(2).mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        total_loss = (self.critic_coef * critic_loss) + actor_loss - (self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return values.mean().item()

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
