import envs.wrappers
import subprocess as sp


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

def environment_wrapper(save_dir, env_name, video_record=False, clip_rewards=True):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """

    
    env = envs.wrappers.make_atari(env_name, full_action_space=True)
    if video_record:
        path = (save_dir / "video" / f"{env.spec.id}_{time.time()}")
        env = gym.wrappers.Monitor(env, path, mode="evaluation")
    env = envs.wrappers.wrap_deepmind(env, scale=True, clip_rewards=clip_rewards) 
    env = envs.wrappers.wrap_pytorch(env)
    return env