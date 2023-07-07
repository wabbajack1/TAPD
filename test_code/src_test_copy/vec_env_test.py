import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from runner import environment_wrapper
import torch
from commons.model import Active_Module, KB_Module, ProgressiveNet

def make_env():
    def _init():
        env = environment_wrapper(None, "StarGunnerNoFrameskip-v4", False)
        return env
    return _init

if __name__ == '__main__':
    env = SubprocVecEnv([make_env() for _ in range(4)])
    state = env.reset()
    states = []
    print(torch.FloatTensor(state).shape)
    active_model = Active_Module("cuda:0", lateral_connections=False).to("cuda:0")
    kb_model = KB_Module("cuda:0").to("cuda:0")
    net = ProgressiveNet(kb_model, active_model).to("cuda:0")
    v, p, _, _ = net(torch.FloatTensor(state).to("cuda:0"))
    action = net.act(torch.FloatTensor(state).to("cuda:0"))
    next_state, reward, done = env.step(action)
    print(v.shape, p.shape, action.shape)
    states.append(torch.FloatTensor(next_state))
    print(len(states[0]))
    print(torch.stack(states).shape)
    print(torch.stack(states)[-1].shape)
    env.close()