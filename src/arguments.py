import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=20,
        help="Batch size"
    )
    
    parser.add_argument(
        "-msp",
        "--max_steps_progress",
        type=int,
        default=3_500_000,
        help="Number of frames for progress phase")
    
    parser.add_argument(
        "-msc",
        "--max_steps_compress",
        type=int,
        default=1000_000,
        help="Number of frames for compress phase (expected to be smaller number than mfp)")

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0007,
        help="Learning rate")

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.99,
        help="gamma discount value")
    
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=0.00001,
        help="epsilin decay for rms optimizer")

    parser.add_argument(
        "-ent",
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy coef value for exploartion")

    parser.add_argument(
        "-cri",
        "--critic_coef",
        type=float,
        default=0.5,
        help="critic coef value")

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="number of workers for running env")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=44,
        help="seed number")
    
    parser.add_argument(
        "-eval",
        "--evaluate",
        type=int,
        default=10,
        help="Run test with #-of episodes; The episodes get avg over the # of episodes provided")
    
    parser.add_argument(
        "-bF",
        "--batch_size_fisher",
        type=int,
        default=32,
        help="Batch size for calculating the estimate of the fisher")
    
    parser.add_argument(
        "-bFnumber",
        "--batch_number_fisher",
        type=int,
        default=100,
        help="Numbers of batches for calculating the estimate of the fisher")
    
    parser.add_argument(
        "-ewcgamma",
        "--ewcgamma",
        type=float,
        default=0.99,
        help="This is the decaying factor of the online-ewc algorithm, where ewcgamma = 1 indicates older tasks are more important than newer one")
    
    parser.add_argument(
        "-ewclambda",
        "--ewclambda",
        type=float,
        default=4000,
        help="The scale at which the regularizer is used (here 175 based on P&C Paper)")
    
    parser.add_argument(
        "-load_step_active",
        "--load_step_active",
        type=int,
        default=0,
        help="From which timestep do you want to load your agent?")
    parser.add_argument(
        "-load_step_kb",
        "--load_step_kb",
        type=int,
        default=0,
        help="From which timestep do you want to load your agent?")
    parser.add_argument(
        "-load_path",
        "--load_path",
        type=str,
        default=None,
        help="Where is the path of your agent?")
    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        default="cpu",
        help="When loading the agent, on which device should it run the evaluation (cpu/gpu)?")
    parser.add_argument(
        "-visits",
        "--visits",
        type=int,
        default=1,
        help="How many times should the tasks visited?")
    parser.add_argument(
        "-ewc_start_timestep_after",
        "--ewc_start_timestep_after",
        type=int,
        default=250000,
        help="Which timestep should the ewc loss be applied to the total loss as a regularizer?")
    parser.add_argument(
        "-n",
        "--notes",
        type=str,
        default="",
        help="Specify the information for this run.")
    
    parser.add_argument(
        "-c",
        "--gpu",
        type=bool,
        default=False,
        help="Mpdels running on GPU.")
    
    args = parser.parse_args()

    return args