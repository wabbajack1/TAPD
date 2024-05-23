# Continual Deep Reinforcement Learning with Task-Agnostic Policy Distillation

## Abstract
The future expectation for universal learning systems is to solve multiple tasks without needing to retrain from scratch when new data arrives. This is because each task requires significant training time. Addressing the problem of continual learning requires the implementation of various methods, given the complexity of the problem space. This problem space includes: (1) addressing catastrophic forgetting to retain previously learned tasks, (2) demonstrating positive forward transfer for faster learning, (3) ensuring scalability across numerous tasks, (4) enabling positive backward transfer, and (5) facilitating learning without requiring task labels, even in the absence of clear task boundaries. In this paper, the Task-Agnostic policy distillation framework is introduced. This framework alleviates problems (1)-(5) by incorporating a task-agnostic phase, where an agent explores its environment without any external goal and maximizes only its intrinsic motivation. The knowledge gained during this phase is later distilled for further exploration. Therefore, the agent acts in a self-supervised manner by systematically seeking novel states. By utilizing task-agnostic distilled knowledge, the agent can solve downstream tasks more efficiently, leading to improved sample efficiency.

![Overview of Variant 1](image.png)

*1. Installation*
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```
Make sure Python 3.8 or higher is installed on your system and that you use a virtual environment of your choice.

When running the code, make sure to use the following command:
```bash
python train.py --config config/example_progress_and_compress.yml
```
Here within the config folder, you can define various configuration files for different experiments.

Additionally the implementation of the baseline might be the first in pytorch: https://arxiv.org/abs/1805.06370.