import tensorflow as tf
import random
import numpy as np
from collections import deque
import time

########################code with gradienttape + grayscale


import sys
import os
# Add the directory containing my_metrics.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utility')))
import MyMetrics


import gym
import procgen
numbers_lvls = 20
env = gym.wrappers.GrayScaleObservation(gym.make('procgen-leaper-v0', 
                #render_mode="human",
                num_levels=numbers_lvls, start_level=1, 
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 ), keep_dim=False)

'''env = gym.make('procgen-bossfight-v0', 
                #render_mode="human",
                num_levels=numbers_lvls, start_level=1, 
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 )'''

#wrappers
#env = gym.wrappers.FrameStack(env, 4)
#env = gym.wrappers.GrayScaleObservation(env)


model_Q = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='valid', activation='relu', input_shape=(64,64,1)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(15, activation='linear')
])


'''model_Q = tf.keras.models.Sequential([
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
])'''


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
model_Q.load_weights("./checkpoints_new/project_3.3_CNN_NEWPAPER_Model_leaper_GRAYSCALE-gradienttape_500episodes_20lvl_rewSTANDARD_sestoround_MonteCarlo_highepsilon_CLIPPING_0001LR.weights.h5")


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
number_episodes = 500
max_number_steps = 1000
learning_rate = 0.0001

epsilon = 0  #prima era 70
discount_epsilon = 0.0003 #prima era 5
lower_bound_epsilon = 0#.40  #prima era 20
###############################
title_save_weights = ("./checkpoints_new/project_3.3_CNN_NEWPAPER_Model_leaper_GRAYSCALE-gradienttape_"
+str(number_episodes)+"episodes_"
+str(numbers_lvls)+"lvl_rewSTANDARD_settimoround_MonteCarlo_highepsilon_CLIPPING_0001LR.weights.h5")
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
    print(env.observation_space)
    average_loss = 0
    average_rew = 0

    #reset temp buffer
    temporary_buffer = []


    m1.startTimer()

    
    print(obs.shape)

    #obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2], 1)

    #predict action probabilities
    print(obs.shape)

    #stacked_images = np.stack([obs[0], obs[1], obs[2]], axis=-1)

    #stacked_images = np.expand_dims(stacked_images, axis=0)

    #action_value = model_Q.predict(stacked_images)

    obs = np.expand_dims(obs, axis=0)

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
            rew= -10
        #elif(done and info['prev_level_complete']==1):
        #    rew= average_rew
        #else:
        average_rew = average_rew + rew #+ 0.1  #0.1 is step reward for surviving
        

        

        #stacked_images_new = np.stack([obs_new[0], obs_new[1], obs_new[2], obs_new[3]], axis=-1)

        #stacked_images_new = np.expand_dims(stacked_images_new, axis=0)

        #action_value_new = model_Q.predict(stacked_images_new)

        obs_new = np.expand_dims(obs_new, axis=0)

        action_value_new = model_Q.predict(obs_new)
        print(action_value_new)



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
        #replay_buffer.append(transition)
        temporary_buffer.append(transition)

        if done:
            obs = env.reset()
            print("Episode finished after {} time steps".format(s+1))
            print(info)
            print(info['prev_level_complete'])
            #if(info['prev_level_complete']==0):
                #average_rew-=20
            for x in temporary_buffer: 
                obs, action_value_new, index_to_update, rew, obs_new, action_value = x 
                new_x = obs, action_value_new, index_to_update, (rew+average_rew), obs_new, action_value
                replay_buffer.append(new_x)
            break
            
        
        obs = obs_new
        action_value = action_value_new

    #print("replay buffer "+str(replay_buffer))

    
    #sampling from replay buffer
    batch_size = 500
    states = np.empty((0, 64, 64), dtype=np.uint8)
    actions = np.empty((0,15),dtype=np.float16)

    for iteration in range(batch_size):
        obs, action_value_new, index_to_update, rew, obs_new, action_value = random.choice(replay_buffer)
        st = np.array(obs)
        at1 = np.array(action_value_new)
        ix = np.array(index_to_update)
        rw = np.array(rew)
        st1 = np.array(obs_new)
        at = np.array(action_value)

        #print(at.shape)
        target = np.copy(at)
        target[0][ix] = rw + gamma*np.max(at1)
        #print("hey")
        #print(target[0][ix])
        #print(states.shape)
        #print(st.shape)

        states = np.concatenate([states, st], axis=0)
        actions = np.concatenate([actions,target], axis=0)
        

    print("dimensions of fit parameters")
    print(states.shape)
    print(actions.shape)


    
    with tf.GradientTape() as tape:
        all_Q_values = model_Q(states, training=True)
        loss = loss_function(actions,all_Q_values) #ground truth values,predicted values

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