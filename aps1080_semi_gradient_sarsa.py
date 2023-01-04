"""
Creates a class FuncApproxAgent, that takes an environment instance as input 
and utilizes the semi-gradient SARSA algorithm to solve the reinforcement 
learning problem for the given environment.

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

In the istance of cartpole used for this problem:
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


"""




import gym
import numpy as np
import keras
import tensorflow as tf
from random import shuffle


class FuncApproxAgent:

  def __init__(self, env, alpha = 0.001, epsilon = 0.2, gamma = 1, epochs = 1, clipnorm = 10, n = 8 ):

    self.alpha = alpha  #learning rate for gradient descent
    self.epsilon = epsilon  #controls exploration vs exploitation
    self.gamma = gamma  #discounting if necessary

    #controlling for n in n-SARSA
    if n>= 1:
      self.n = n
    else:
      self.n = 1

    self.current_model_to_update = 0  #ID of model who's gradient is being computed this episode
    # 0 corresponds to the main model
    # 1 corresponds to the model initially used to compute the targets
    # This value is updated at the end of each episode after checking done
    self.num_state_features = len(env.reset())
    self.action_space = env.action_space


    #Temporary information for each episode
    self.states_seen_in_episode = list() #[index is timestep ]
    self.rewards_seen_in_episode = list() #[index is timestep ]
    self.actions_taken_in_episode = list() #[index is timestep ]

    
    self.clipnorm = clipnorm #Limit gradient to avoid excessive changes in action value function -> Inf or NaN
    self.reg_factor = 0.001
    self.optimizer = keras.optimizers.Adam(learning_rate = self.alpha, clipnorm = self.clipnorm)
    self.target_optimizer = keras.optimizers.Adam(learning_rate = self.alpha, clipnorm = self.clipnorm)
    self.replay_memory = list() 
    self.epochs = epochs
    self.model, self.target_model = self._buildNN()




  def _buildNN(self):
    
    #Create two similar neural networks
    #One generates the gradients, one generates the target
    #input is state, output is an action value for each action possible
    model_input = keras.layers.Input(shape = ((self.num_state_features),)) #Input is features (4) 
    h1 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(model_input) 
    h2 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(h1)
    h3 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(h2)
    model_output = keras.layers.Dense(self.action_space.n,  activation = "linear")(h3) #Output is action values for each action

    model = keras.Model(inputs = model_input, outputs = model_output)



    t_model_input = keras.layers.Input(shape = ((self.num_state_features),)) #Input is features (4) 
    t_h1 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(t_model_input) 
    t_h2 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(t_h1)
    t_h3 = keras.layers.Dense(16, activation = "relu", kernel_regularizer=keras.regularizers.l2(self.reg_factor))(t_h2)
    t_model_output = keras.layers.Dense(self.action_space.n,  activation = "linear")(t_h3) #Output is action values for each action

    t_model = keras.Model(inputs = t_model_input, outputs = t_model_output)

    return model, t_model




  def reset(self):
      #Clear out recorded State-Action-Reward Tuples from the episode
      self.states_seen_in_episode = list() #[index is timestep ]
      self.rewards_seen_in_episode = list() #[index is timestep ]
      self.actions_taken_in_episode = list() #[index is timestep ]



  def policy_action(self, current_state):
      #Epsilon greedy policy 

      chosen = np.random.rand() #Generate a probability from 0 to 1
      
      #If below epsilon take a random action, otherwise take the known best action
      if chosen <= self.epsilon:
        return self.action_space.sample()

      predictions = self.model( current_state.reshape(1,self.num_state_features)  )
      predictions = predictions[0]
      #print("Action Values: ", predictions)
      return np.argmax(predictions)





  def manual_grad_update(self, state, action, discounted_returns, update_target = 0):
    #Apply Semi Gradient SARSA update method
    
    #Create a mask for the action values
    model_input = np.array(state).reshape((1,self.num_state_features))
    one_hot_action = np.zeros(shape = (1,self.action_space.n))
    one_hot_action[0][action] = 1

    if update_target != 1:
      with tf.GradientTape() as tape:
        #compute q_s_a and its gradient in the form [0, q_s_a]
        predictions = self.model(model_input, training = True)

        #Mask the action values using the selected action
        q_s_a_vector = predictions[0] * one_hot_action
        q_s_a = tf.math.reduce_sum(q_s_a_vector)
        
        #Compute the gradient of the active action value prediction
        grads = tape.gradient(q_s_a,self.model.trainable_weights)

        #compute [G - q_s_a] where G = R
        #(Scale the action value by the delta between return and action value)
        factor = discounted_returns - q_s_a.numpy()
        factor = (-1) * factor
        factor = float(factor)

    
        norm_sum = 0
        over_lim = False
        #multiply to obtaing [R- q_s_a]grad(q_s_a)
        for index in range(len(grads)):
          grads[index] = factor * grads[index]

          #Check if gradient might be getting too big
          norm = tf.norm(grads[index])
          norm = float(norm)
          norm_sum+= norm**2
          if norm >= self.clipnorm:
            over_lim = True

        #Limit gradient if too large
        if over_lim == True:
          for index in range(len(grads)):
            norm = tf.norm(grads[index])
            grads[index] = grads[index] * (self.clipnorm)/(norm_sum**2)
          
          over_lim = False
    
      #Perform the gradient descent update
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    elif update_target == 1:
      with tf.GradientTape() as tape:
        #compute q_s_a and its gradient in the form [0, q_s_a]
        predictions = self.model(model_input, training = True)

        q_s_a_vector = predictions[0] * one_hot_action
        q_s_a = tf.math.reduce_sum(q_s_a_vector)
        grads = tape.gradient(q_s_a,self.model.trainable_weights)

        #compute [G - q_s_a] where G = R
        factor = discounted_returns - q_s_a.numpy()
        factor = (-1) * factor
        factor = float(factor)

        norm_sum = 0
        over_lim = False
        #multiply to obtaing [R- q_s_a]grad(q_s_a)
        for index in range(len(grads)):
          grads[index] = factor * grads[index]

          #Check if gradient might be getting too big
          norm = tf.norm(grads[index])
          norm = float(norm)
          norm_sum+= norm**2
          if norm >= self.clipnorm:
            over_lim = True

        #Limit gradient if too large
        if over_lim == True:
          for index in range(len(grads)):
            norm = tf.norm(grads[index])
            grads[index] = grads[index] * (self.clipnorm)/(norm_sum**2)
          
          over_lim = False
        


      self.target_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    return








  def _semi_grad_SARSA_base(self, new_n = None):
      #Updates n steps back using SARSA
      #This function is implemented by semi_grad_SARSA_update which handles
      #generating the lookback window for the action values of the states
      #at the end of each episode
      
      if new_n != None:
        backstep = new_n
      else:
        backstep = self.n

      index = len(self.actions_taken_in_episode) - backstep - 1
      end_index = index + backstep

      if index >= 0:
        state = self.states_seen_in_episode[index] 
        action = self.actions_taken_in_episode[index]
        reward = self.rewards_seen_in_episode[index]

        #If we have only the pre terminal state, the return is the last reward
        if backstep == 0:
          self.manual_grad_update(state, action, reward, self.current_model_to_update)

          return


        #Compute G

        discounted_rewards = np.zeros(shape = backstep)
        
        if backstep < self.n:
          #If we have less than n steps after the state we sum all the remaining rewards
          for i in range(backstep):
            discounted_rewards[i] = self.rewards_seen_in_episode[index + i] *(self.gamma ** i)
          
          discounted_returns = np.sum(discounted_rewards)
        else:
          #If we have the n steps available we sum n rewards
          #and use the action-value function for the end state
          #to approximate the returns
          for i in range(backstep):
            discounted_rewards[i] = self.rewards_seen_in_episode[index + i] *(self.gamma ** i)
          
          end_state =  self.states_seen_in_episode[end_index] 
          end_action = self.actions_taken_in_episode[end_index]
          
          model_input = np.array(end_state).reshape((1,self.num_state_features))
          if self.current_model_to_update == 0:
            predictions = self.target_model(model_input)
          elif self.current_model_to_update == 1:
            predictions = self.model(model_input)
          q_s_end_a_end = predictions[0][end_action]
          discounted_returns = np.sum(discounted_rewards) + ((self.gamma ** self.n) * q_s_end_a_end)


        #update the action value NN weights
        self.manual_grad_update(state, action, discounted_returns, self.current_model_to_update)

        #Record the transition for updating target generation neural network
        if (len(self.replay_memory) > 1000):
          self.replay_memory.remove(self.replay_memory[0])
        
        replay = [state, action, discounted_returns]
        self.replay_memory.append(replay)
        

        return




  def semi_grad_SARSA_update(self, done):
      #Updates n steps back using Semi Gradient SARSA

      if done == True:
        #If we are done all episodes run the last full update then run 
        #SG SARSA on the last n states with the returns as sum of rewards
        for i in range(self.n + 1):
            self._semi_grad_SARSA_base(new_n = self.n - i)
      else:
          self._semi_grad_SARSA_base()
      
      return








  def run_step(self, current_state, action = None, reward = None, done = False):
         
      #Add the state to our lists
      self.states_seen_in_episode.append(np.array(current_state)) #record the full state for function approx 

      if action != None:
        self.actions_taken_in_episode.append(action)
        #action that was taken 

      if reward != None:
        self.rewards_seen_in_episode.append(reward)
        #reward that was received from taking action

      self.semi_grad_SARSA_update(done)

        #print(self.model.trainable_weights)
      if done == True:
        #If we are done we update the model that calculates targets with our replay memory data

        for epoch in range(self.epochs):
          
          update_list = self.replay_memory.copy()
          shuffle(update_list)
          for (c_state, c_action, c_discounted_returns) in update_list:
            #print("Updating Target Model")
            self.manual_grad_update(c_state, c_action, c_discounted_returns, update_target = 1 - self.current_model_to_update)

        self.current_model_to_update = 1 - self.current_model_to_update



      return current_state

env = gym.make("CartPole-v0")
env.max_episode_steps = 12000
max_episodes = 1000

episode_lengths = np.zeros(shape = max_episodes)

agent = FuncApproxAgent(env = env, alpha = 0.001, epsilon = 1, gamma = 1, epochs = 2, clipnorm = 50, n = 8 )

for episode in range(max_episodes):
  agent.epsilon = 0.995*agent.epsilon #Adding some decay for each episode
  obs = env.reset()
  c_state = agent.run_step(obs)

  for i in range(10000):
    action = agent.policy_action(c_state)
    obs, reward, done, info = env.step(action)
    c_state = agent.run_step( obs, action, reward, done)
  

    if done:
      agent.reset()
      break
  episode_lengths[episode] = i
  print("episode: ", episode, "length: ", episode_lengths[episode], "\n" )

env.close()

agent.model.predict(np.array([ 0.03102564,  0.01683985, -0.03239316, -0.00890066]).reshape((1,4))     )

print(episode_lengths)

print(np.max(episode_lengths))

print(np.average(episode_lengths[:100]))
print(np.average(episode_lengths[-100:]))
