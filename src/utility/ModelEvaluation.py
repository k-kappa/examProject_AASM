import tensorflow as tf
import random
import numpy as np
import MyEvaluation

'''
import sys
import os
# Add the directory containing my_metrics.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utility')))
import MyMetrics'''


import gym
import procgen
'''env = gym.make('procgen-bossfight-v0', #env for coinrun
                #render_mode="human",
                num_levels=0, start_level=50, 
                distribution_mode='hard', 
                use_backgrounds=True, 
                rand_seed=7858 )'''
env = gym.wrappers.GrayScaleObservation(gym.make('procgen-leaper-v0', #wrapper for bossfight and chaser
                #render_mode="human",
                num_levels=0, start_level=50, 
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 ), keep_dim=False)


model_Q = tf.keras.models.Sequential([ ###network for bossfight DQN + remember grayscale wrapper
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

title_save_weights = "./project_3.3_CNN_NEWPAPER_Model_leaper_GRAYSCALE-gradienttape_500episodes_20lvl_rewSTANDARD_settimoround_MonteCarlo_highepsilon_CLIPPING_0001LR.weights.h5"
#load weights if necessary
model_Q.load_weights(title_save_weights) ######path to weights
loss_fn = tf.keras.losses.MeanSquaredError()
model_Q.compile(optimizer='adam', loss= loss_fn)


#definition of our policy 
def policy(listOptions,epsilon):
    sample = random.uniform(0,1) #sample the epsilon probability
    if sample < epsilon: #select random action
        return random.randint(0,15)
    else: #select action according the policy
        return np.argmax(listOptions)
        

#Hyperparameters
number_episodes = 500
max_number_steps = 1500
epsilon = 0
flagFrameStack = False  # set to true if the network uses framestack wrapper, and remember to change the input size of the network to (64,64,3), otherwise for other games it is always (64,64,1)
###############################

if flagFrameStack:
    env = gym.wrappers.FrameStack(env, 3)


eval =MyEvaluation.MyEvaluation(number_episodes)




rew = 0
counter_wins=0

for episode in range(number_episodes): #for each episode...
    print("episode number "+ str(episode))
    obs = env.reset() #reset environment and get observation
    #print(obs)
    #print(obs.shape)
    average_loss = 0
    average_rew = 0


    reasonEnd = -1

    for s in range(max_number_steps): #for each step...


        if flagFrameStack:
            stacked_images = np.stack([obs[0], obs[1], obs[2]], axis=-1)

            stacked_images = np.expand_dims(stacked_images, axis=0)

            action_value = model_Q.predict(stacked_images)
        else:
            obs = np.expand_dims(obs, axis=0)
            #predict action probabilities
            #print(obs.shape)
            action_value = model_Q.predict(obs)
            #print("action value shape")
            #print(action_value.shape)

        ######obs = np.squeeze(obs, axis=0)

        #choose action following the policy
        action = policy(action_value[0], 0)
        print("choosen action: "+str(action))
        obs_t = obs #save previous observation

        #take action and get observations
        obs, rew, done, info = env.step(action)
        #print("fresh rew: "+str(rew))
        #rew = rew -1
        average_rew = average_rew + rew



        if done:
            obs = env.reset()
            print("Episode finished after {} time steps".format(s+1))
            if(s == max_number_steps):
                print("endless")
                reasonEnd=2
            elif(rew==10):
                print("win")
                reasonEnd=0
            else:
                print("loose")
                reasonEnd=1
            if(info['prev_level_complete']==1):
                counter_wins+=1
            break
        eval.addInfo(reasonEnd,episode)
        eval.addReward(rew,episode)
        eval.incrementStep(episode)
    eval.addInfo(reasonEnd,episode)
    eval.addReward(average_rew,episode)

    print("episode "+ str(episode)+" got reward "+ str(average_rew))

eval.addWins(counter_wins)
eval.printRealPlayer()

#########################################################
# Now is the turn of the random player

eval.randomPlayer(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
counter_wins_random=0
for episode in range(number_episodes): #for each episode...
    print("episode number "+ str(episode))
    obs = env.reset() #reset environment and get observation
    #print(obs)
    #print(obs.shape)
    average_loss = 0
    average_rew = 0


    reasonEnd = -1

    for s in range(max_number_steps): #for each step...


        obs = np.expand_dims(obs, axis=0)
        #predict action probabilities
        #print(obs.shape)
        action = eval.randomPlay(episode)

        obs_t = obs #save previous observation

        #take action and get observations
        obs, rew, done, info = env.step(action)
        #print("fresh rew: "+str(rew))
        #rew = rew -1
        average_rew = average_rew + rew



        if done:
            obs = env.reset()
            print("Episode finished after {} time steps".format(s+1))
            if(s == max_number_steps):
                print("endless")
                reasonEnd=2
            elif(rew==10):
                print("win")
                reasonEnd=0
            else:
                print("loose")
                reasonEnd=1
            if(info['prev_level_complete']==1):
                counter_wins_random+=1
            break
        eval.addRandomInfo(reasonEnd,episode)
        eval.addRandomReward(rew,episode)

    eval.addInfo(reasonEnd,episode)
    eval.addRandomReward(average_rew,episode)

eval.addWinsRandom(counter_wins_random)
print(eval.str()) #print the results



env.close()