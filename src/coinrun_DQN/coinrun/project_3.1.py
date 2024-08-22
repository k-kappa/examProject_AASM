import tensorflow as tf
import random
import numpy as np
from collections import deque
import time


import sys
import os
# Add the directory containing my_metrics.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utility')))
import MyMetrics


import gym
import procgen
numbers_lvls = 15
env = gym.make('procgen-coinrun-v0', 
                render_mode="human",
                num_levels=numbers_lvls, start_level=4,
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 )


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
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(15)
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
model_Q.load_weights("./checkpoints_new/project_3.1_CNN_NEW_Model_COINRUN500episodes_15lvl_rewBasic_primoround_restartepsilon.weights.h5")


loss_fn = tf.keras.losses.MeanSquaredError()
#model_Q.compile(optimizer='adam', loss= loss_fn)


#definition of our policy 
def policy(listOptions,epsilon):
    sample = random.uniform(0,1) #sample the epsilon probability
    if sample < epsilon: #select random action
        return random.randint(0,15)
    else: #select action according the policy
        return np.argmax(listOptions)
        

#Hyperparameters
alpha= 0.9
gamma= 0.95
number_episodes = 100
max_number_steps = 1000
learning_rate = 0.001

epsilon = 0.70  #prima era 70
discount_epsilon = 0.0002 #prima era 5
lower_bound_epsilon = 0.30  #prima era 20
###############################
title_save_weights = ("./checkpoints_new/project_3.1_CNN_NEW_Model_COINRUN"
+str(number_episodes)+"episodes_"
+str(numbers_lvls)+"lvl_rewBasic_secondoround_restartepsilon.weights.h5")
#print(title_save_weights)


m1 =MyMetrics.MyMetrics(number_episodes)

optimizer = tf.keras.optimizers.Adam(learning_rate)
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


    m1.startTimer()

    obs = np.expand_dims(obs, axis=0)
    #predict action probabilities
    #print(obs.shape)
    action_value = model_Q.predict(obs)
      
    action = policy(action_value[0], epsilon-(episode*discount_epsilon))
    print("choosen action: "+str(action))

    #startREW = time.time()

    for s in range(max_number_steps): #for each step...

        #take action and get observations
        obs_new, rew, done, info = env.step(action)
        #print(info)
        if(rew!=0):
            print("---->"+str(rew))

        if(done and info['prev_level_complete']==0):  ################changed
                rew-=10
        #if(time.time()-startREW>=1000):
            #rew = rew + 1
            #startREW=time.time()

        #rew = rew + 0.05
        average_rew = average_rew + rew

        obs_new = np.expand_dims(obs_new, axis=0)

        action_value_new = model_Q.predict(obs)

        if(epsilon-(episode*discount_epsilon)<=lower_bound_epsilon):
            epsilon_to_apply = lower_bound_epsilon
        else:
            epsilon_to_apply = epsilon-(episode*discount_epsilon)



        #epsilon_to_apply=lower_bound_epsilon#########################remember to change
        action = policy(action_value[0], epsilon_to_apply)
        print("choosen action: "+str(action))
        print("With epsilon: "+str(epsilon_to_apply))
        m1.addEpsilon(epsilon_to_apply,episode)
        

        #print(action_value.shape)
        index_to_update = np.argmax(action_value) #index of previous action value to update


        #Store the transition in the replay buffer
        transition = (obs, action_value_new, index_to_update, rew, obs_new, action_value)
        replay_buffer.append(transition)


        if done:
            obs = env.reset()
            print("Episode finished after {} time steps".format(s+1))
            print(info)
            print(info['prev_level_complete'])

            break
        
        obs = obs_new
        action_value = action_value_new

    #print("replay buffer "+str(replay_buffer))

    
    #sampling from replay buffer
    batch_size = 500
    states = np.empty((0, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((0,15),dtype=np.float16)

    for iteration in range(batch_size):
        random_sample = random.choice(replay_buffer)
        st = np.array(random_sample[0])
        at1 = np.array(random_sample[1])
        ix = np.array(random_sample[2])
        rw = np.array(random_sample[3])
        st1 = np.array(random_sample[4])
        at = np.array(random_sample[5])

        #print(at.shape)
        target = np.copy(at)
        target[0][ix] = rw + gamma*np.max(at1)
        #print("hey")
        #print(target[0][ix])

        states = np.concatenate([states, st], axis=0)
        actions = np.concatenate([actions,target], axis=0)
        

    print("dimensions of fit parameters")
    print(states.shape)
    print(actions.shape)

    with tf.GradientTape() as tape:
        all_Q_values = model_Q(states, training=True)
        loss = loss_function(actions,all_Q_values) 

    # Calcolare i gradienti
    gradients = tape.gradient(loss, model_Q.trainable_variables)

    # Applicare i gradienti utilizzando l'ottimizzatore
    optimizer.apply_gradients(zip(gradients, model_Q.trainable_variables))

    #training of the network
    #training = model_Q.fit(states, actions, epochs=1,verbose=2)
    print("episode "+ str(episode)+" got reward "+ str(average_rew))

    m1.endTimer()
    m1.addLoss(loss,episode)
    m1.addReward(average_rew,episode)
    m1.addDuration(m1.getDuration(),episode)

model_Q.summary()

# Save the weights
model_Q.save_weights(title_save_weights)

m1.saveMetrics()
m1.showAllGraphs()
print("Training duration: "+str(time.time()-start))



env.close()
