from new_a2c.stable_baselines3.common.envs.bit_flipping_env import BitFlippingEnv
from new_a2c.stable_baselines3.common.envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from new_a2c.stable_baselines3.common.envs.multi_input_envs import SimpleMultiObsEnv

__all__ = [
    "BitFlippingEnv",
    "FakeImageEnv",
    "IdentityEnv",
    "IdentityEnvBox",
    "IdentityEnvMultiBinary",
    "IdentityEnvMultiDiscrete",
    "SimpleMultiObsEnv",
    "SimpleMultiObsEnv",
]
