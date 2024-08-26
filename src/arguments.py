import argparse
import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='progress-compress', help='algorithm to use: progress-compress | ewc-online | progressive-nets | agnostic (make the agnostic-phase flag True)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.95,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.001,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=10,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=20,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=5e4,
        help='eval steps')
    parser.add_argument(
        '--num-env-steps-progress',
        type=int,
        default=1.5e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--num-env-steps-compress',
        type=int,
        default=2e5,
        help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--num-env-steps-agnostic',
        type=int,
        default=300_000,
        help='number of environment steps to train after sampling randomly.')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help='enable mps | cuda training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--visits',
        type=int,
        default=1,
        help='How many visits to one task.')
    parser.add_argument(
        '--batch-size-fisher',
        type=int,
        default=32,
        help='How many batches to calculate the fisher information')
    parser.add_argument(
        '--steps-calculate-fisher',
        type=int,
        default=100,
        help="How many times batches are sampled from the environment during computation of the Fisher importance")
    parser.add_argument(
        '--ewc-start-timestep-after',
        type=int,
        default=1e5,
        help='When to start the calucation of fisher estimation')
    parser.add_argument(
        '--ewc-gamma',
        type=float,
        default=0.3,
        help='decay factor (i.e. 1 does not forget old task)')
    parser.add_argument(
        '--ewc-lambda',
        type=int,
        default=200,
        help='How large should we consider the ewc penalty')
    parser.add_argument(
        '--agnostic-phase',
        type=bool,
        default=False,
        help='Enable agnostic phase')
    
    parser.add_argument(
        '--log_wandb',
        type=bool,
        default=False,
        help='Enable logging to wandb')
    
    parser.add_argument(
        "--agn_samples",
        type=int,
        default=5,
        help="Number of samples in agnostic phase")
    parser.add_argument(
        "--model_path_active",
        type=str,
        default=None,
        help="Path to model for active learning")
    
    parser.add_argument(
        "--model_path_kb",
        type=str,
        default=None,
        help="Path to model for kb learning")
    
    parser.add_argument(
        '--config', type=str, help="Path to configuration YAML file", default=None
    )
    
    args = parser.parse_args()

    # Set device between cuda and mps
    assert args.device in ["cuda", "mps", "cpu"]

    # Load YAML config if provided
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

        # Overwrite default args with YAML config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args
