# examProject_AASM

Repo structure:

--in the folder "src" there is all the code wrote for the exam, organized by game and algorithm(DQN or Actor Critic). All files have similar structure, they differ in the logic part: such as the rewards management, hyperparameters, network architectures and env wrapper used. To be able to understand the code written for the project it is sufficient to consult the "baseline_code_version", there are all the learning versions used(of course they are not identical to each game.To explore a specific approach used for a game, please follow the following path-rule:
(ex. coinrun with DQN) 
src->coinrun_DQN->coinrun->project_3.3.py

-the main games faced were: coinrun colorized, bossfight grayscaled, chaser grayscaled. It is possible to find code of other games but they are not officialy reported beacuse not exploited too much(ex. heist,maze,starpilot,fruitbot)

--the folder "baseline_code_version" contains file that don't differ too much:
project_3.1.py = DQN with replay buffer and TD(0) approach
project_3.2.py = DQN with replay buffer and Monte Carlo approach, with step penalty
project_3.22.py = DQN with replay buffer and Monte Carlo approach, with no step penalty
project_3.222.py = same as 3.1 , duplicate just for self-mind-organization 
project_3.3.py = DQN with replay buffer and Monte Carlo approach, with no step penalty, grayscale and framestack wrappers
project_5.0_actorCritic.py = it is the version fo Actor Critic Algorithm with a TD(0)* approach

-in each game folder is possible to see graphs of the trainings

--the staff for evaluation can be found in src/utility , where ModelEvaluation.py had the goal to evaluate the networks. It is possible to find the saved DQN weights of each game 




*with TD(0) approach it is meant there is no reward redistribution, so each step gets the reward that gains in the game, while Monte Carlo approach propagate the reward/average_reward of the episode to all the steps.
