# Basic RL survey
- https://jonathan-hui.medium.com/rl-introduction-to-deep-reinforcement-learning-35c25e04c199

# Policy interation vs value iteration

+ Policy iteration: After applying policy evaluation (one iteration == going through all states and update based on Bellman expectation equation as an update function) and the convergance (:= some threshold values of the update values) of that (with respect ot number of iteration or discount factor) we update our policy with the help of poliy improvment (greedy based method on state values)
  - retaining the previous policy’s converged state values when starting the next policy evaluation 
  - [Remark](https://towardsdatascience.com/policy-and-value-iteration-78501afb41d2#:~:text=In%20Policy%20Iteration%2C%20at%20each,be%20the%20estimated%20state%20value): Policy Iteration takes an initial policy, evaluates it, and then uses those values to create an improved    policy. These steps of evaluation and improvement are then repeated on the newly generated policy to give an even better policy. This process continues until, eventually, we end up with the optimal policy.


+ Value iteration: Make a policy imporvemnt after each policy iteration and not after we have converged with respect to the changes in the state values.
  - combines the Policy Evaluation and Policy Improvement stages into a single update.
  - Differnce to policy iteration: The only difference is, in the original Policy Evaluation equation, the next state value was given by the sum over the policy’s probability of taking each action, whereas now, in the Value Iteration equation, we simply take the value of the action that returns the largest value.

see also this [link](https://medium.com/analytics-vidhya/bellman-equation-and-dynamic-programming-773ce67fc6a7).



# Stuff to do it my own / read
- [x] [Knowledge Distillation](https://towardsdatascience.com/model-distillation-and-compression-for-recommender-systems-in-pytorch-5d81c0f2c0ec)
- [x] [DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [x] [actor critic network](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [x] [REINFORCE](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63) "“actor” part of Actor-Critic methods"
- [ ] [Progressive Nets](https://towardsdatascience.com/progressive-neural-networks-explained-implemented-6f07366d714d#a1b0)
- [ ] [DDPG](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) RL in continues action spaces
- [x] [MAP](https://gregorygundersen.com/blog/2019/05/08/laplaces-method/)
- [x] [Gaussian Process](https://distill.pub/2019/visual-exploration-gaussian-processes/#:~:text=A%20multivariate%20Gaussian%20distribution%20has,distribution%20is%20also%20N%20%2Ddimensional.) note => gaussian process != gaussian approximation

# Fisher Info
- [x] [Fisher(1)][https://andrewliao11.github.io/blog/fisher-info-matrix/]; [Fisher(2)](https://towardsdatascience.com/an-intuitive-look-at-fisher-information-2720c40867d8); [Fisher(3)](https://awni.github.io/intro-fisher-information/#eq:fisher_information)
- My notes on [Fisher](https://hypernotes.zenkit.com/i/Yi9T5ZeN9O/VWMHo4IvYND/?v=dJvC65TVGc)

# Notes for understanding rl domain

## How is supervised learning related to reinforcment learning?

Supervised learning and reinforcement learning (RL) are both types of machine learning, but they are used to solve different types of problems and they have different characteristics.

Supervised learning is a type of machine learning where an algorithm is trained on a labeled dataset, where the correct output or label for each input is provided. The goal of supervised learning is to learn a mapping from inputs to outputs, or to classify inputs into different categories. Supervised learning is typically used for problems such as image classification, speech recognition, or natural language processing.

On the other hand, reinforcement learning (RL) is a type of machine learning where an agent interacts with an environment and learns to take actions that maximize a reward signal. The agent learns through trial and error by observing the consequences of its actions and receiving feedback in the form of rewards or penalties. RL is typically used for problems such as control systems, decision-making or game playing.

In summary, supervised learning is used to learn a mapping from inputs to outputs, it requires labeled data and the goal is to generalize to unseen data. RL is used to learn a policy that maximizes a reward signal, it requires interaction with an environment and the goal is to optimize the policy.

## So how are the target label connected?

In supervised learning, the target label is the correct output or label that the algorithm is trying to predict for a given input. The algorithm is trained on a labeled dataset, where the correct output or label for each input is provided. The goal of supervised learning is to learn a mapping from inputs to outputs, or to classify inputs into different categories.

In reinforcement learning (RL), the target label is the reward signal that the agent is trying to maximize. The agent interacts with an environment and learns to take actions that maximize the reward signal. The agent learns through trial and error by observing the consequences of its actions and receiving feedback in the form of rewards or penalties.

So, in supervised learning, the target label is the correct output or label that the algorithm is trying to predict, while in RL, the target label is the reward signal that the agent is trying to maximize.

## So is the ground truth label in rl the bellman equation?

The Bellman equation is a fundamental equation in reinforcement learning (RL) that is used to update the action-value function or Q-value of an agent. It is not directly related to the concept of a ground truth label in supervised learning.

The Bellman equation defines the optimal action-value function as the expected sum of future rewards, given the current state and action. It expresses the relationship between the value of a state-action pair and the value of the next state and the reward received.

## The bellman eqation can be seen as the correct label like in supervised learning?

The Bellman equation in reinforcement learning (RL) can be thought of as a way to update the action-value function or Q-value of an agent, similar to how a ground truth label is used in supervised learning to update the model's predictions.

In supervised learning, the ground truth label is the correct output or label that the algorithm is trying to predict for a given input. The algorithm is trained on a labeled dataset, where the correct output or label for each input is provided. The goal of supervised learning is to learn a mapping from inputs to outputs, or to classify inputs into different categories.

In RL, the agent interacts with an environment and learns to take actions that maximize a reward signal. The Bellman equation is used to update the Q-value of a state-action pair based on the expected sum of future rewards, given the current state and action. It expresses the relationship between the value of a state-action pair and the value of the next state and the reward received. The Q-value is updated to reflect the agent's estimated future rewards.

So, in a sense, the Bellman equation can be seen as a way to update the Q-value based on the agent's estimated future rewards, similar to how a ground truth label is used in supervised learning to update the model's predictions based on the correct output.

It's also worth noting that, in RL the ground truth is not available as the future is uncertain and the agent needs to explore to discover the optimal policy, while in supervised learning the ground truth is available

## The ground truth in rl needs to be approximated and is not certain?

In RL, the ground truth is not certain, it needs to be approximated through the agent's interaction with the environment and the agent's estimates of the optimal policy improve over time as the agent gains more experience. In contrast, in supervised learning, the ground truth is known and provided in the form of labeled data.

## Is the discounted reward a random variable? [vid](https://www.youtube.com/watch?v=lI8_p7Qeuto)
- G_t = R_t + γ * R_{t+1} + γ^2 * R_{t+2} + ... + γ^{T-t} * R_T

- Current estimated value at current state s and action a: Q(s; a)

- *Estimated* value at next state s’ if then performing a’: Q(s’; a’)

- Bellman equation allows a better estimate of the current value: Q(s; a) = r + *𝛾·Q(s`; a`)*; the bold expr. is from the bellman eq. the target label (like in supervised learning) -- we use this bold expr. to update our paramters/value functions

- The TD error (rather: the square of it; compare with the L2 norm) is usable like a cost function to update the parameters of a function (nerual net) that estimates Q given s and a as its inputs: Q(s; a) ← Q(s; a) + η·*(r + 𝛾·Q(s`; a`) - Q(s; a))* the bold expr. is the td-error

- Q(s; a) tells if taking the action a at state s is good or not (like a critic) 

- In summary depends Q(s; a) on s, a, A_t+1 ~ pi(.|s_t) (policy) and S_t+1 ~ p(.|s, a) (state transtion density func.) -- S_t, A_t are random variables we sample from

- The state value function V(s) just says how good we ware in the current situation (avarage performance); E[V(s)] evaluates how good pi (policy is) i.e. objective function of *policy based learning*

The discounted return is a measure of the total discounted future rewards an agent can expect to receive for a given state and action. It is defined as the sum of the discounted rewards for all time steps starting from the current time step.

It can be seen as a random variable, because if we would know the G_t than we would know if someone is close to G_t (basically see if someone is losing or winning a game).

The discounted return is used to evaluate the current policy and is used to update the Q-value or value function

So for evaluation of our current policy we would use E[G_T|s_t, a_t].





# Model based vs model free methods

## Model free
Find policy or value function directly
+ "Tools/Algos":
	+ [Policy gradient methods](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63); see also the great post about [policy gradient](http://karpathy.github.io/2016/05/31/rl/)
	  - used in actor part of Actor Critic Methods
    - Policy gradient methods update the probability distribution of actions so that actions with higher expected reward have a higher probability value for an observed state

## Model based
Find the transiton probability of the env (p(s'|a, s)) directly

# Proposed architecture
> Remark: Column := neural net; block := layer
+ P & C method for learning
+ active column (actor-critc network) & knowledge based network (actor-critc network)
+ Two columns (i.e. active column & knowledge based network) like in Progressive Nets. Both networks are connected via lateral layer (adaptors).
+ EWC as the penalty term in the distillation process
+ Curiosity-driven Exploration ICM module (only foward model) in active column [and knowledge based network].

Steps to build self supervised task agnostic progress and compress reinforcment agent:
1. Build P & C architecture
  -  Implment two columns both are Advantage Actor Critic (A2C) networks, sharing the output layer.
  -  For Implementing use [doric framework](https://github.com/arcosin/Doric) *test if you can build it with that*
2. Implement EWC (as a class)
3. ICM Module (only forward model) onto active column
3. Training logic, i.e. switching between task and activating compress phase. Consider also switching between different loss function during progress phase (here use forward model) and in compress phase use KL divergence + EWC. Indicate also in which phase the agent is, e.g. print("active phase") ... print("compress phase")

# Tests

| *Hyperparameter*      	| *Permuted Mnist* 	|   	|   	|   	|
|---------------------	|----------------	|---	|---	|---	|
| learning rate       	| 1e-3           	|   	|   	|   	|
| n. hidden layers    	| 2              	|   	|   	|   	|
| width hidden layers 	| 400            	|   	|   	|   	|
| epochs / dataset    	| 20             	|   	|   	|   	|


## Test1: group of numbers
+ 1. Setting: 
  - Two tasks: predict the two number groups [(1, 2, 3, 4), (5, 6, 7, 8, 9)]
  - two columns(kb, active)
  - reset for active is off
  - Trained for 10 Epochs
  - no ewc penalty in compress phase
  - distillation is still on in compress phase
  - Ran active column on test set (10 Trials), avg over all 10 Trials:
    - stats = [83.136856, 28.454845],
              [56.493008, 64.05197 ]]
  - Conclusion: The Network seems to have remembered the old task and gained positve Transfer, because of the lateral connection of the kb column, which were trained during the destilation phase, i.e. the kb transfered the knowledge via the lateral connection to the active column which are only trained during the progress phase. This suggests that the active column uses neurons in the hidden layer (which are frozen) of the kb column, which are specilized in encoding specific shapes/patterns in the image for prediction and trains only the lateral connections.

+ 2. Setting:
  - Same setting as 1., but ran kb column on test set (5x10 Trials), avg over all 10 Trials of the tensor, i.e. played 5 sets of 10 games:
    - stats  = [[76.80171 , 28.429634],
                [57.388107, 61.25547 ]]
    - Conclusion: The kb column seems to have remembered the old task and used old positve transfered knowledge. This means at first the kb column uses the distilled knoweldge of the first task for prediction, which is not suprising. But if the kb column predicts the second task, then it uses encoded knowledge of its prevoius self, because the active column uses during training weights of the kb column (specilized neurons for shapes are frozen) to minimize its loss via lateral connections which get trained. This means that old weights help to minimize the loss, which in return gets distilled again in the kb column. So basically feeds the kb column a more abstract reprentation of its old knowledge in its self over tasks via a buffer (the active column) which transforms/stretches the information of old tasks (this knowledge represention might get less accurate (vanish) with more tasks, hence we should use ewc)

+ 3. Setting:
  - 10 epochs
  - Only one network trained, without protection and P&C framework
  - no reset of networks after one task
  - stats: [92.06543 , 30.697712],
            [53.414078, 84.11113 ]]

+ 4. Setting:
  - same as 3., but with reset on
  - stats with 10 epochs: [90.44807 , 30.389507],
                          [44.38911 , 86.99407 ]]
  - stats with 100 epochs: [97.99999 , 31.430927],
                           [45.86564 , 97.590935]]


## Test2: Permuted MNIST