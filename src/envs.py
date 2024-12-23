import os

import gymnasium as gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_


class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservations, self).__init__(env)

    def observation(self, observation):
        return observation / 255.0
    
def environment_mapping_action_all(env):
    # Create a one-to-one mapping from 0 to 17 for all possible actions
    action_map = {
        0: 0,    # NOOP
        1: 1,    # FIRE
        2: 2,    # UP
        3: 3,    # RIGHT
        4: 4,    # LEFT
        5: 5,    # DOWN
        6: 6,    # UPRIGHT
        7: 7,    # UPLEFT
        8: 8,    # DOWNRIGHT
        9: 9,    # DOWNLEFT
        10: 10,  # UPFIRE
        11: 11,  # RIGHTFIRE
        12: 12,  # LEFTFIRE
        13: 13,  # DOWNFIRE
        14: 14,  # UPRIGHTFIRE
        15: 15,  # UPLEFTFIRE
        16: 16,  # DOWNRIGHTFIRE
        17: 17,  # DOWNLEFTFIRE
    }

    # Apply the action map to the environment using the UnifiedActionWrapper
    env = UnifiedActionWrapper(env, action_map=action_map)
    return env

def environment_mapping_action(env):
    
    if "Pong" in env.spec.id:
        # "Pong" already has the desired action space
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3,  # LEFT
        }
    if "BeamRider" in env.spec.id:
        # Adjusted action maps for "BeamRider" 
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 3,  # RIGHT
            3: 4,  # LEFT
        }
    if "SpaceInvaders" in env.spec.id:
        # Adjusted action maps for "SpaceInvaders"
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3   # LEFT
        }
    if "DemonAttack" in env.spec.id:
        # Adjusted action maps for "DemonAttack"
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3   # LEFT
        }
    if "AirRaid" in env.spec.id:
        # Adjusted action maps for "AirRaidNoFrameskip"
        action_map = {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # RIGHT
            3: 3   # LEFT
        }

    unified_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    env = UnifiedActionWrapper(env, action_map=action_map, UNIFIED_ACTIONS=unified_actions)
    return env

class UnifiedActionWrapper(gym.Wrapper):

    def __init__(self, env, action_map, **kwargs):
        super(UnifiedActionWrapper, self).__init__(env)
        self.action_map = action_map

        # Get the UNIFIED_ACTIONS from kwargs, if provided
        unified_actions = kwargs.get('UNIFIED_ACTIONS')
        
        # Set the action space based on UNIFIED_ACTIONS or default to 18 actions
        self.action_space = gym.spaces.Discrete(len(unified_actions) if unified_actions is not None else 18)

    def step(self, action):
        return self.env.step(self.action_map[np.array(action).item()])
    

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dmc2gym.make(domain_name=domain, task_name=task)
            env = ClipAction(env)
        else:
            env = gym.make(env_id, full_action_space=False)

        # is_atari = hasattr(gym.envs, 'atari') and isinstance(
        #     env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            # if args is not None:
            #     env = gym.wrappers.RecordVideo(env, os.path.join(args.save_dir, args.algo), episode_trigger = lambda x: (x+1) % 2 == 0)
            #     print(env)
            ...
        
        env = Monitor(env,os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
            
        # if len(env.observation_space.shape) == 3:
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=84, height=84)
        env = ClipRewardEnv(env)
        env = environment_mapping_action(env)
        # env = environment_mapping_action_all(env)
        env = ScaledFloatFrame(env)


        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]


    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=True)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_obs=True)
    
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=False)

    print("Normalized reward")

    envs = VecPyTorch(envs, device)

    print("Stack frame:", num_frame_stack, envs.observation_space.shape)
    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)


    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, _, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if isinstance(ob, tuple):
            ob, _ = ob

        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        # obs = torch.unsqueeze(obs, 1)
        # print(obs.shape)
        # obs = obs.permute(1, 0, 2, 3)
        # print(obs.shape)
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        # print(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs

        return self.stacked_obs

    def close(self):
        self.venv.close()
