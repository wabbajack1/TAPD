import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from src.storage import RolloutStorage
from collections import deque

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.backends.mps.is_available() and use_cuda:
        t = t.to("mps:0")
    return Variable(t, **kwargs)

class A2C():
    def __init__(self,
                 big_policy,
                 actor_critic,
                 forward_model,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):
        
        self.big_policy = big_policy
        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop((list(actor_critic.parameters()) + list(big_policy.adaptor.parameters())), lr, eps=eps, alpha=alpha)
        self.optimizer_forward = optim.RMSprop(forward_model.parameters(), lr, eps=eps, alpha=alpha)
        # self.optimizer_forward = optim.RAdam(forward_model.parameters(), 0.001)


    def update(self, rollouts, fwd_losses=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # print(rollouts.obs[:-1].view(-1, *obs_shape).shape)
        # print(rollouts.obs[:-1].view(-1, *obs_shape).shape)

        values, action_log_probs, dist_entropy = self.big_policy.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape)
        )
        
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()

        # print(sum(fwd_losses).sum())
        # print(value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)

        if fwd_losses is not None:
            self.optimizer_forward.zero_grad()
            ((value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)).backward()
            torch.cat(fwd_losses).mean().backward()
            self.optimizer_forward.step()
        else:
            (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
    

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

class Distillation():
    def __init__(self,
                 actor_critic_teacher,
                 actor_critic_student,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic_teacher = actor_critic_teacher
        self.actor_critic_student = actor_critic_student
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.ewc_loss = torch.tensor(0)

        self.optimizer = optim.RMSprop(actor_critic_student.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts, ewc):
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        total_loss = 0

        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        _, action_log_probs_teacher, _ = self.actor_critic_teacher.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape)
        )

        _, action_log_probs_student, _ = self.actor_critic_student.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape)
        )


        action_log_probs_teacher = action_log_probs_teacher.view(num_steps, num_processes, 1)
        action_log_probs_student = action_log_probs_student.view(num_steps, num_processes, 1)

        kl_loss = criterion(action_log_probs_student, action_log_probs_teacher.detach())

        if ewc is not None and ewc.ewc_timestep_counter >= ewc.ewc_start_timestep:
            self.ewc_loss = ewc.penalty(self.actor_critic_student)
            total_loss = kl_loss + self.ewc_loss
        else:
            total_loss = kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic_student.parameters(),self.max_grad_norm)

        self.optimizer.step()

        return total_loss.item(), kl_loss.item(), self.ewc_loss.item()

class EWConline(object):
    def __init__(self, entropy_coef=0.01, ewc_lambda=175, ewc_gamma=0.3, ewc_start_timestep_after=80_000, max_grad_norm=None, steps_calucate_fisher=None):
        """The online ewc algo. We need current samples from the current policy (in atari domain task == env == data)
        Args:
            task (None): the task (in atari a env) for calculating the importance of task w.r.t the paramters
            model (nn.Module): the model which params are important to protect
            ewc_gamma (float, optional): the deacay factor. Defaults to 0.4.
            device (_type_, optional): _description_. Defaults to None.
        """
        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor
            
        self.ewc_timestep_counter = 0
        self.ewc_start_timestep = ewc_start_timestep_after
        self.ewc_gamma = ewc_gamma
        self.ewc_lambda = ewc_lambda
        self.entropy_coef = entropy_coef
        self.old_fisher = None
        self.mean_params = {}
        self.max_grad_norm = max_grad_norm
        self.steps_calucate_fisher = steps_calucate_fisher
        self.exp = 0

    def empty_fisher(self):
        self.fisher = {}
        for n, p in deepcopy(self.params).items():
            self.fisher[n] = variable(p.detach().clone().zero_())
    
    def calculate_fisher(self, rollouts):
        self.model.eval()
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape)
        )
        
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.model.zero_grad()
        (action_loss - dist_entropy * self.entropy_coef).backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)

        # calculate Fisher information matrixc
        # y_t = a * y_t + (1-a)*y_{t-1}
        for name, param in self.model.named_parameters():
            if param.grad is not None and "critic" not in name:
                self.fisher[name] += param.grad.data.clone().pow(2) / self.steps_calucate_fisher
    
    def get_fisher(self):
        self.exp += 1 # count tasks
        self.normalize_fisher() # normalize the fisher after calculation of importance of the new task
        if self.exp > 1:
            for name, param in self.model.named_parameters():
                if param.grad is not None and "critic" not in name:
                    if self.old_fisher is not None and name in self.old_fisher:
                        self.fisher[name] = self.ewc_gamma * self.old_fisher[name] + self.fisher[name]
                        # print(f"{name} - decay", "->", f"min {self.fisher[name].min()}, median {self.fisher[name].median()}, max {self.fisher[name].max()}")
                        # print("---")
                        # print(f"{name} - decay", "->", f"min {self.old_fisher[name].min()}, median {self.old_fisher[name].median()}, max {self.old_fisher[name].max()}")
        
        self.model.train()
        return deepcopy(self.fisher)
    
    def normalize_fisher(self):
        for name in self.fisher:
            # self.fisher[name] = (self.fisher[name] - self.fisher[name].mean()) / (self.fisher[name].std()  + 1e-20)
            self.fisher[name] = (self.fisher[name] - self.fisher[name].min()) / (self.fisher[name].max() - self.fisher[name].min()  + 1e-20) + 1e-5
            # print(name, "->", f"min {self.fisher[name].min()}, median {self.fisher[name].median()}, max {self.fisher[name].max()}")

    
    def penalty(self, model: nn.Module):
        """Calculate the penalty to add to loss.

        Args:
            ewc_lambda (int): the lambda value
            model (nn.Module): The model which gets regulized (its the model, which traines and gets dynamically updated)

        Returns:
            _type_: float
        """

        # integrate old fisher into new fisher and after calulation, make normalization
        loss = 0
        for n, p in model.named_parameters():
            if "critic" not in n:
                # print(n, "->", f"min {self.fisher[n].min()}, median {self.fisher[n].median()}, max {self.fisher[n].max()}")
                # fisher = torch.sqrt(self.fisher[n] + 1e-08)
                fisher = self.fisher[n]
                loss += (fisher * (p - self.mean_params[n]).pow(2)).sum()
                # fisher_sum += abs(self.fisher[n]).sum()
                # mean_params_sum += self.mean_params[n].sum()
        
        # print(n, self.mean_params[n])
        # print(self.ewc_lambda * loss)
        return self.ewc_lambda * loss
    
    @torch.no_grad()
    def update_parameters(self, model, env_name):
        """Update the model, after learning the latest task. Here we calculate
        directly the FIM and also reset the mean_params.

        Args:
            agent: to get the new data (experience) of the latest run from the agents memory (current policy)
            new_task (_type_): _description_
        """
        self.model = model
        self.env_name = env_name
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad and "critic" not in n}
        # print(list(self.params.keys())[-1], list(self.params.values())[-1])
        self.empty_fisher()
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data.detach().clone())
        print(f"Calculation of the task for the importance of each parameter: {self.env_name}")


def gather_fisher_samples(current_policy, ewc, args, envs, device):
    """_summary_

    Args:
        current_policy (_type_): The current policy to calculate the estimates of the fisher (here always the kb network!)
        ewc (_type_): The ewc object to calculate the ewc fisher (here like update)
        active_agent (_type_): _description_
        args (_type_): _description_
        envs (_type_): _description_
        device (_type_): _description_
        env_name (_type_): _description_
    """
    rollouts = RolloutStorage(args.batch_size_fisher, args.num_processes, envs.observation_space.shape, envs.action_space)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs/255.0)
    rollouts.to(device)
    episode_rewards = deque(maxlen=100)

    for _ in range(args.steps_calucate_fisher):
        # nmb of steps (rollouts) before update
        for step in range(args.batch_size_fisher):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = current_policy.act(rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs = obs/255.0

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)
        

        with torch.no_grad():
            next_value = current_policy.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        # gradient update
        ewc.calculate_fisher(rollouts)
        rollouts.after_update()
    ewc.old_fisher = ewc.get_fisher()
