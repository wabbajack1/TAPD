import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./model_arch')

class A2C_ACKTR():
    def __init__(self,
                 big_policy,
                 actor_critic,
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

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
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

        self.optimizer = optim.RMSprop(actor_critic_student.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        writer.add_graph(self.actor_critic_teacher, rollouts.obs[:-1].view(-1, *obs_shape))
        writer.close()

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

        self.optimizer.zero_grad()
        kl_loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic_student.parameters(),self.max_grad_norm)

        self.optimizer.step()

        return kl_loss.item()


# from copy import deepcopy
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable
# import torch.utils.data
# from typing import Optional
# from commons.memory.CustomDataset import CustomDataset
# from torch.utils.data.dataloader import DataLoader
# from torch.nn.utils import parameters_to_vector

# def variable(t: torch.Tensor, use_cuda=True, **kwargs):
#     if torch.cuda.is_available() and use_cuda:
#         t = t.cuda()
#     return Variable(t, **kwargs)

# class EWC(object):
#     def __init__(self, agent:None, model: nn.Module, ewc_lambda=175, ewc_gamma=0.4, batch_size_fisher=32, ewc_start_timestep_after=1000, device=None, env_name:Optional[str] = None):
#         """The online ewc algo 
#         Args:
#             task (None): the task (in atari a env) for calculating the importance of task w.r.t the paramters
#             model (nn.Module): the model which params are important to protect
#             ewc_gamma (float, optional): the deacay factor. Defaults to 0.4.
#             device (_type_, optional): _description_. Defaults to None.
#         """
#         if torch.cuda.is_available():
#             self.FloatTensor = torch.cuda.FloatTensor
#         else:
#             self.FloatTensor = torch.FloatTensor
            
#         self.ewc_start_timestep = ewc_start_timestep_after
#         self.model = model
#         self.device = device
#         self.ewc_gamma = ewc_gamma
#         self.ewc_lambda = ewc_lambda
#         self.env_name = env_name
#         self.agent = agent # we need the memory module of this object (in atari domain task == env == data)
#         self.batch_size_fisher = batch_size_fisher

#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad and "critic" not in n}
#         # self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self.mean_params = {}
#         self.old_fisher = None
#         self.fisher = self.calculate_fisher() # calculate the importance of params for the previous task
#         self.mean_model = deepcopy(self.model)
        
#         for n, p in deepcopy(self.params).items():
#             self.mean_params[n] = variable(p.data)
    
#     def calculate_fisher(self):
#         print(f"Calculation of the task for the importance of each parameter: {self.env_name}")
#         self.model.train()
        
#         fisher = {}
#         for n, p in deepcopy(self.params).items():
#             fisher[n] = variable(p.detach().clone().zero_())
            

#         for states, actions, true_values in dataloader:
#             #print("Parameters:", len(dataset), len(dataloader), states.shape, len(dataloader)/self.agent.no_of_workers, self.batch_size_fisher)
#             # print("batch size ewc", states.shape, actions.shape, true_values.shape)
            
#             # Calculate gradients
#             self.model.zero_grad()
            
#             states = states.to(self.device)
#             actions = actions.to(self.device)
#             true_values = true_values.to(self.device)
#             values, log_probs, entropy = self.model.evaluate_action(states, actions)
            
#             values = torch.squeeze(values)
#             log_probs = torch.squeeze(log_probs)
#             entropy = torch.squeeze(entropy)
#             true_values = torch.squeeze(true_values)
            
#             advantages = true_values - values
#             # critic_loss = advantages.pow(2).mean()
            
#             actor_loss = -(log_probs * advantages.detach()).mean()
#             # total_loss = ((0.5 * critic_loss) + actor_loss - (0.01 * entropy)).backward()
#             actor_loss.backward() # calc the gradients and store it in grad
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
#             # Update Fisher information matrix
#             # y_t = a * y_t + (1-a)*y_{t-1}
#             for name, param in self.model.named_parameters():
#                 if param.grad is not None and "critic" not in name:
#                     if self.old_fisher is not None and name in self.old_fisher:
#                         fisher[name] += self.ewc_gamma * self.old_fisher[name] + param.grad.detach().clone().pow(2)
#                     else:
#                         fisher[name] += param.grad.detach().clone().pow(2)

#         print(f"len of dataloader {len(dataloader)}, no of workers {self.agent.no_of_workers}")
#         for name in fisher:
#             # fisher[name].data = (fisher[name].data - torch.mean(fisher[name].data).detach()) / (torch.std(fisher[name].data).detach() + 1e-08)
#             fisher[name].data /= len(dataloader)/self.agent.no_of_workers
        
#         self.old_fisher = fisher.copy()
#         self.model.train()
        
#         return fisher
    
#     def penalty(self, model: nn.Module):
#         """Calculate the penalty to add to loss.

#         Args:
#             ewc_lambda (int): the lambda value
#             model (nn.Module): The model which gets regulized (its the model, which traines and gets dynamically updated)

#         Returns:
#             _type_: float
#         """
#         loss = 0
#         fisher_sum = 0
#         mean_params_sum = 0
#         for n, p in model.named_parameters():
#             if "critic" not in n:
#                 # fisher = torch.sqrt(self.fisher[n] + 1e-08)
#                 fisher = self.fisher[n]
#                 loss += (fisher * (p - self.mean_params[n]).pow(2)).sum()
#                 fisher_sum += abs(self.fisher[n]).sum()
#                 mean_params_sum += self.mean_params[n].sum()
#                 # print(n, torch.sqrt(self.fisher[n] + 1e-05))
        
#         print("EWC Loss", (self.ewc_lambda * loss).item(), "loss fisher", loss.item(), f"EWC lambda {self.ewc_lambda}", f"Fisher: {fisher_sum.sum()}", f"mean params: {mean_params_sum.sum()}")
#         return self.ewc_lambda * loss
    
#     def update(self, agent, model, env_name):
#         """Update the model, after learning the latest task. Here we calculate
#         directly the FIM and also reset the mean_params.

#         Args:
#             agent: to get the new data (experience) of the latest run from the agents memory (current policy)
#             model (_type_): _description_
#             new_task (_type_): _description_
#         """
#         self.agent = agent
#         self.env_name = env_name
#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad and "critic" not in n}
#         self.fisher = self.calculate_fisher()
#         for n, p in deepcopy(self.params).items():
#             self.mean_params[n] = variable(p.data)
            
            
# # def compute_distance(model1, model2, mode="euclidean"):
# #     params1 = parameters_to_vector(model1.parameters())
# #     params2 = parameters_to_vector(model2.parameters())

# #     if mode == "euclidean":
# #         return torch.norm(params1 - params2).item()
# #     elif mode == "cosine":
# #         return torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
# #     else:
# #         raise ValueError(f"Unknown mode: {mode}")