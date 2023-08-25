import gym

UNIFIED_ACTIONS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

class UnifiedActionWrapper(gym.Wrapper):

    def __init__(self, env, action_map):
        super(UnifiedActionWrapper, self).__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.Discrete(len(UNIFIED_ACTIONS))

    def step(self, action):
        return self.env.step(self.action_map[action])