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
'''env = gym.make('procgen-coinrun-v0', #env for coinrun
                render_mode="human",
                num_levels=0, start_level=50, 
                distribution_mode='hard', 
                use_backgrounds=True, 
                rand_seed=7858 )'''
env = gym.wrappers.GrayScaleObservation(gym.make('procgen-bossfight-v0', #wrapper for starpilot and leaper
                render_mode="human",
                num_levels=0, start_level=50, 
                distribution_mode='easy', 
                use_backgrounds=False, 
                rand_seed=7585 ), keep_dim=False)




#definition of our policy 
def policy(listOptions):
    sample = random.uniform(0, 1)  # Sample a probability value between 0 and 1
    counter = -1
    for x in listOptions:
        counter += 1
        if sample <= x:  # Check if the sample falls within the current probability range
            return counter
        sample -= x
        

#Hyperparameters #################################
gamma= 0.95
number_episodes = 500
max_number_steps = 1500
learning_rate_actor = 0.00001
#learning_rate_critic = 0.00001
flagFrameStack = True  # set to true if the network uses framestack wrapper, and remember to change the input size of the network to (64,64,3), otherwise for other games it is always (64,64,1)
###############################

optimizer_actor = tf.keras.optimizers.Adam(learning_rate_actor)
#optimizer_critic = tf.keras.optimizers.Adam(learning_rate_critic)
loss_function = tf.keras.losses.MeanSquaredError()

if flagFrameStack:
    env = gym.wrappers.FrameStack(env, 4)


eval =MyEvaluation.MyEvaluation(number_episodes)

oi = tf.keras.initializers.Orthogonal()

model_Actor = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='same', activation='relu', input_shape=(4, 64, 64)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=oi),
    tf.keras.layers.Dense(15, activation='softmax')

    
])

model_Actor.load_weights("./project_4.01_CNN_NEW_SMALL_Model_bossfight_ACTORCRITIC_500episodes_20lvl_pureTD(0)(1-1)-nosteppenality_lastAttempt_DYNAMIC_terzoround_00001LR_ACTOR.weights.h5")


rew = 0
counter_wins=0

for episode in range(number_episodes): #for each episode...
    print("episode number "+ str(episode))
    obs = env.reset() #reset environment and get observation
    #print(obs)
    #print(obs.shape)
    average_rew = 0

    #reset temp buffer
    #temporary_buffer = []
    loss_critic_cumulative_episode = 0
    loss_actor_cumulative_episode = 0

    

    obs = np.expand_dims(obs, axis=0)
    #predict action probabilities
    #print(obs.shape)
    action_probabilities = model_Actor.predict(obs)
      
    action = policy(action_probabilities[0])
    print("choosen action: "+str(action))
    FinalRew=0

    #startREW = time.time()
    steps_made=0

    for s in range(max_number_steps): #for each step...

        # Predict action probabilities and state value
        action_probabilities = model_Actor(obs/255, training=True)
        #state_value = model_Critic(obs/255, training=True)

        # Choose an action based on policy
        print(action_probabilities[0])
        action = policy(action_probabilities[0])
        print("Chosen action: " + str(action))

        # Take action and get new observation
        obs_new, rew, done, info = env.step(action)
        obs_new = np.expand_dims(obs_new, axis=0)

        average_rew += rew


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
        #eval.addInfo(reasonEnd,episode)
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