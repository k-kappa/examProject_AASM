import tensorflow as tf
import random
import numpy as np
from collections import deque
import time
import math

########################code with gradienttape


import sys
import os
# Add the directory containing my_metrics.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utility')))
import MyMetrics


import gym
import procgen
numbers_lvls =  1
env = gym.make('procgen-coinrun-v0', 
                render_mode="human",
                num_levels=numbers_lvls, start_level=4, 
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 )




'''
model_Q = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='valid', activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(15, activation='linear')
])'''

oi = tf.keras.initializers.Orthogonal()

model_Critic = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model_Actor = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu',kernel_initializer=oi),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(15, activation='softmax')
])


'''
model_Q = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(15)
])'''

'''
model_Q = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (64, 64, 3)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(9)
])'''

'''
model_Q = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (64, 64, 3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(9)
])'''
'''
model_Q = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape = (64, 64, 3)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(9)
])'''


#load weights if necessary
#model_Q.load_weights("./checkpoints_new/project_3.22_CNN_NEW_Model_heist500episodes_20lvl_rewEPISODIC-nosteppenality_quartoround_restartepsilonwherestopped_0003LR.weights.h5")


loss_fn = tf.keras.losses.MeanSquaredError()
#model_Q.compile(optimizer='adam', loss= loss_fn)


#definition of our policy 
def policy(listOptions):
    sample = random.uniform(0, 1)  # Sample a probability value between 0 and 1
    counter = -1
    for x in listOptions:
        counter += 1
        if sample <= x:  # Check if the sample falls within the current probability range
            return counter
        sample -= x
        

#Hyperparameters
gamma= 0.95
number_episodes = 50
max_number_steps = 3000
learning_rate_actor = 0.00001
learning_rate_critic = 0.00001

epsilon = 0.7#70  #prima era 70
discount_epsilon = 0.0003 #prima era 5
lower_bound_epsilon = 0.35  #prima era 20
###############################
title_save_weights = ("./checkpoints_new/project_4.1_CNN_NEW_Model_coinrun_ACTORCRITIC_"
+str(number_episodes)+"episodes_"
+str(numbers_lvls)+"lvl_rewEPISODIC(1-1)-batch_primoround_restartepsilon_0005LR.weights.h5")
#print(title_save_weights)



m1 =MyMetrics.MyMetrics(number_episodes)

optimizer_actor = tf.keras.optimizers.Adam(learning_rate_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate_critic)
loss_function = tf.keras.losses.MeanSquaredError()



rew = 0

# Initialize replay buffer with a max size of 10000
replay_buffer = deque(maxlen=10000)



start = time.time()

for episode in range(number_episodes): #for each episode...
    print("episode number "+ str(episode))
    obs = env.reset() #reset environment and get observation
    #print(obs)
    #print(obs.shape)
    average_loss = 0
    average_rew = 0

    #reset temp buffer
    temporary_buffer = []
    
    loss_critic_cumulative_episode = 0
    loss_actor_cumulative_episode = 0


    m1.startTimer()

    obs = np.expand_dims(obs, axis=0)
    #predict action probabilities
    #print(obs.shape)
    #action_probabilities = model_Actor.predict(obs)
    #print(action_probabilities[0])
      
    #action = policy(action_probabilities[0])
    #print("choosen action: "+str(action))
    FinalRew=0

    #startREW = time.time()
    steps_made=0

    for s in range(max_number_steps): #for each step...


        '''with tf.GradientTape(persistent=True) as tape:
            #take action and get observations
            obs_new, rew, done, info = env.step(action)
            #print(info)
            if(rew!=0):
                print("---->"+str(rew))


            obs_new = np.expand_dims(obs_new, axis=0)


            average_rew = average_rew + rew #+ 0.05

            #predict state values
            state_value_new = model_Critic.predict(obs_new)
            state_value = model_Critic.predict(obs)

        
        
            delta = rew + gamma*state_value_new*(1-done) + state_value
            loss_actor = -tf.math.log(action_probabilities[0][action]) * delta
            loss_critic = tf.square(delta)


        loss_actor_cumulative_episode+=loss_actor
        loss_critic_cumulative_episode+=loss_critic'''

        
        # Predict action probabilities and state value
        action_probabilities = model_Actor.predict(obs/255)
            

        # Choose an action based on policy
        print(action_probabilities[0])
        sum=0
        for x in action_probabilities[0]:
            sum+=x
        print(sum)
        action = policy(action_probabilities[0])
        print("Chosen action: " + str(action))

        # Take action and get new observation
        obs_new, rew, done, info = env.step(action)
        obs_new = np.expand_dims(obs_new, axis=0)

        #rew-=1 #step penality

        if(done and info['prev_level_complete']==1): 
            FinalRew=10
        elif(done and info['prev_level_complete']==0):
            FinalRew=-10



        average_rew += rew

        #loss_actor_cumulative_episode+=loss_actor
        #loss_critic_cumulative_episode+=loss_critic
        #print(loss_actor_cumulative_episode)

        #Store the transition in the replay buffer
        transition = (rew,obs,obs_new,done)
        temporary_buffer.append(transition)


        if done:
            #obs = env.reset()
            print("Episode finished after {} time steps".format(s+1))
            print(info)
            print(info['prev_level_complete'])
            steps_made=s
            for x in temporary_buffer: #for each step of the game give the gained reward(10 if total win, 1 if semi-win,-1 if loose)
                rew,obs,obs_new,done = x 
                new_x = rew+FinalRew,obs,obs_new,done #########################deccay the rewards#######################################
                replay_buffer.append(new_x)
            break

        
        obs = obs_new

    


    print("episode "+ str(episode)+" got reward "+ str(average_rew))

    #sampling from replay buffer
    for iteration in replay_buffer:
        rew,obs,obs_new,done = random.choice(replay_buffer)
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(model_Actor.trainable_variables)
            #tape.watch(model_Critic.trainable_variables)

            action_probabilities = model_Actor(obs/255, training=True)

            state_value = model_Critic(obs/255, training=True)
            state_value_new = model_Critic(obs_new/255, training=True)

            delta = rew + gamma * state_value_new * (1 - done) - state_value

            # Calculate losses
            loss_actor = -tf.math.log(action_probabilities[0][action]) * delta
            loss_critic = tf.square(delta)
            #print(loss_actor)
            #print(loss_critic)
            m1.addLossActorCritic(loss_actor.numpy()[0][0],loss_critic.numpy()[0][0],episode)

        gradients_actor = tape.gradient(loss_actor, model_Actor.trainable_variables)
        gradients_critic = tape.gradient(loss_critic, model_Critic.trainable_variables)
        optimizer_actor.apply_gradients(zip(gradients_actor, model_Actor.trainable_variables))
        optimizer_critic.apply_gradients(zip(gradients_critic, model_Critic.trainable_variables))
    
    del tape
    m1.endTimer()
    #m1.addLossActorCritic(loss_actor_cumulative_episode/steps_made,loss_critic_cumulative_episode/steps_made,episode)
    m1.addReward(average_rew,episode)
    m1.addDuration(m1.getDuration(),episode)


# Save the weights
model_Actor.save_weights(title_save_weights+"_ACTOR.weights.h5")
# Save the weights
model_Critic.save_weights(title_save_weights+"_CRITIC.weights.h5")

m1.saveMetrics()
m1.showAllGraphs()
print("Training duration: "+str(time.time()-start))





env.close()