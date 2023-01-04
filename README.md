# Cart-Pole-SARSA
Appliction of semi-gradient SARSA to solve the cartpole-v0 problem


Creates a class FuncApproxAgent, that takes an environment instance as input 
and utilizes the semi-gradient SARSA algorithm to solve the reinforcement 
learning problem for the given environment. The enviroment must have a 
discrete action space.

Two neural networks are used instead of one in an attempt to improve stability
One neural network generates the target, and another for the gradient in the 
gradient descent. The target generating neural network is trained at the end
of each episode using a record of updates that contains the last 1000 State-
-Action-Reward sets. The two are then alternated.

Note, both neural networks take in a state and output a number of action values
equal to the size of the action space of the environment. 

Hyper parameters are:
    alpha - learning_rate (or lr) for optimizers and gradient descent
    epsilon - exploration rate
    epochs -  iterations on the whole record set for the target generating NN
    clipnorm - the max L2 norm we accept for the gradient; gradients above this
               are scaled down to this size in an effort to prevent explosions
    n - the number of lookback steps used for SARSA

In the instance of cartpole used for this problem:
    alpha - below 0.0005 was found to lead to good function.
    epsilon - I applied a decaying exploration from 1, with a multiplier of 
              0.9995, llowing the algorithm to build up over 200 epsiodes with
              great exploration that approached cartpole v0's default max steps
              fairly quickly.
    epochs - 2 epochs provided a good trade-off between time training and NN 
             accuracy.
    clipnorm - 5, 10, 20 and 50 were used. 20 and 50 were found to be
               sufficiently effective rates.
    n - A default of 8 steps was used for all runs.





