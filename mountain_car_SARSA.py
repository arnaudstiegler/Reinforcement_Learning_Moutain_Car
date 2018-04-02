import numpy as np
import gym
import math

# the discount factor
discount = 0.995;

# The learning rate
alpha = 0.3

# The exploration rate
epsilon = 0.3

# The number of episodes used to evaluate the quality of a policy
n_episodes = 5000;

# maximum number of steps of a trajectory
max_steps = 499;

# number of grids
n_slices = 20

env = gym.make('MountainCar-v0')
env.max_episode_steps = max_steps


n_a = env.action_space.n

#Creating the center for the discretization of position
number_of_center_pos = 20
centers_pos = [0]*number_of_center_pos
sigma_pos = 0.1
for i in range(number_of_center_pos):
    centers_pos[i] = (-1.2 + i*(1.9/(number_of_center_pos-1)))

#Creating the center for the discretization of speed
number_of_center_speed = 15
centers_speed = [0]*number_of_center_speed
sigma_speed = 0.01
for i in range(number_of_center_speed):
    centers_speed[i] = (-0.07 +  i*(0.14/(number_of_center_speed-1)))

#Defining weights table
weigths = np.zeros((3*(number_of_center_pos+number_of_center_speed)+1)) # The +1 is for the bias term
Q = np.zeros(())

# l_pos = len(centers_pos)
# l_speed = len(centers_speed)
# for i in range(l_pos):
#     weigths[i + 2*(l_pos+l_speed)] = 1
# for i in range(l_speed):
#     weigths[i + l_pos + 2*(l_speed+l_pos)] = 1

def rbf_function(s,i,centers,sigma):
    return np.exp(-math.sqrt((centers[i] - s)**2)/(2*sigma))

def phi_function(s,a):
    l_pos = len(centers_pos)
    l_speed = len(centers_speed)
    output = np.zeros(((l_pos+l_speed)*3+1))
    for i in range(l_pos):
        output[i+a*(l_pos+l_speed)] = rbf_function(s[0],i,centers_pos,sigma_pos)
    for j in range(l_speed):
        output[j+l_pos+a*(l_pos+l_speed)] = rbf_function(s[1],j,centers_speed,sigma_speed)
    output[output.shape[0]-1] = 1 #This is for the bias term
    return output


def find_action(s):
    left = phi_function(s, 0).T.dot(weigths)
    none = phi_function(s, 1).T.dot(weigths)
    right = phi_function(s, 2).T.dot(weigths)
    act = max(left,right,none)
    if(act == left):
        return 0
    elif(act == none):
        return 1
    else:
        return 2

def eps_search(observation):
    if np.random.random() < epsilon:
        return 0 if np.random.random() > 2. / 3. else (1 if np.random.random() > 0.5 else 2)
    return find_action(observation)

env = gym.make('MountainCar-v0')


maxPos = -100
minPos = 100
for _ in range(20000):
    observation = env.reset()

    s = observation
    action = eps_search((observation[0],observation[1]))

    done = False

    while(done == False):
        s_Prime, reward, done, info = env.step(action)

        if done and s_Prime[0] >= 0.5:
            reward = 1

        if(s_Prime[0] > maxPos):
            maxPos = s_Prime[0]
        if (s_Prime[0] < minPos):
            minPos = s_Prime[0]

        action_Prime = eps_search((s_Prime[0], s_Prime[1]))

        weigths = weigths + alpha*(reward + discount*phi_function(s_Prime,action_Prime).T.dot(weigths) - phi_function(s,action).T.dot(weigths))*(phi_function(s,action))
        norm = np.linalg.norm(weigths)
        weigths = weigths/norm

        s = s_Prime
        action = action_Prime
    if(_ % 100 == 0):
        print(_)

np.save("weigths", weigths)
print("max: "+ str(maxPos))
print("min Position: "+ str(minPos))
