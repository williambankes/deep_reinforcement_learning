# drlAgents

An implementation of deep reinforcement learning algorithms.

## Example

The following code is an example of how the package can be used to train an agent on the Cartpole environment. The default hyperparameter values are setup to return good results on this environment.

```
from drlAgents import DQNAgent
import matplotlib.pyplot as plt
import gym
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 20
        self.out_features = 2
        
        self.in_layer = nn.Linear(4,a)
        self.h1 = nn.Linear(a,a)
        self.bn1 = nn.BatchNorm1d(a)
        self.h2 = nn.Linear(a,a)
        self.bn2 = nn.BatchNorm1d(a)
        self.out_layer = nn.Linear(a,2)
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        return self.out_layer(x)

env = gym.make('CartPole-v0')
agent = DQNAgent(env, Net)   
agent.train(epochs=300)

agent.play(length=100)
```

## Results and Comments:

The results of training the dqn agent on the cartpole environment can be seen in the gif below. Tuning hyperparameters to ensure maximum performance was the most time consuming element of this process. The gif below shows the results of training the dqn agent using the setup shown in the example above. The agent is successful for the first 200 or so steps and fails after this, leaning to the right.

![Cartpole trained with a dqn agent](https://github.com/williambankes/drlAgents/blob/master/figures/cartpole.gif?raw=true)

## Comparison of DQN and DDQN algorithms:

The below graph shows a comparison of the dqn and ddqn algorithms on the Cartpole environment. Plotted is the mean rewards per epoch/episode of 10 runs on varying starting conditions. This was done to better understand how the algorithm performed despite it's high variance across starting conditions of the same environment. 

![Comparison of mean reward for dqn and ddqn](https://github.com/williambankes/drlAgents/blob/master/figures/dqn_vs_ddqn_2.png?raw=true)
