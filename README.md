# drlAgents

An implementation of deep reinforcement learning algorithms.

## Example

```
policy_params = {
        'discount':0.999,
        'learning_rate': 0.0001,
        'momentum':0.5,
        'use_learning_decay':True,
        'lr_decay':1
        }

agent_params = {
        'min_mem':10_000,
        'max_mem':50_000,
        'batch_size':128,
        'target_update':10,
        }

env_params = {
        'env_string':'CartPole-v0',
        'seed':42}

expl_params = {'eps_start':0.9,
               'eps_end':0.1,
               'eps_decay':0.995}


env = gym.make('CartPole-v0')
p = dqnPolicy(Net, create_factor_min_exploration, expl_params,
              **policy_params)
d = dqnAgent(env, p, **agent_params)

d.train(200)
```
## Requirements:
- torch
- numpy
- random
(TODO: provide exact versionning)

## Results and Comments:

The results of training the dqn agent on the cartpole environment can be seen in the gif below. Tuning hyperparameters to ensure maximum performance was the most time consuming element of this process. The gif below shows the results of training the dqn agent using the setup shown in the example above. The agent is successful for the first 200 or so steps and fails after this, leaning to the right.

![Cartpole trained with a dqn agent](https://github.com/williambankes/drlAgents/blob/master/figures/cartpole.gif?raw=true)

## Comparison of DQN and DDQN algorithms:

The below graph shows a comparison of the dqn and ddqn algorithms on the Cartpole environment. Plotted is the mean rewards per epoch/episode of 10 runs on varying starting conditions. This was done to better understand how the algorithm performed despite it's high variance across starting conditions of the same environment. 

![Comparison of mean reward for dqn and ddqn](https://github.com/williambankes/drlAgents/blob/master/figures/dqn_vs_ddqn_2.png?raw=true)
