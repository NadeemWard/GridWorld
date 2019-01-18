import Grid
from Transitions import Transitions_Probs as tp
import pandas as pd
from Rewards import Reward
from Agent import Agent
import numpy as np
import Value_estimation as ve


# def convert(action):
#     '''
#     Convert an action from a string to its index in the matrix
#     :param action: the string action
#     :return: the index for this action in the matrix
#     '''
#
#     return actions.index(action)

def epsilon_greedy_policy(state_action_estimates, epsilon,actions):


    num_actions , num_states = state_action_estimates.shape
    #print("\n")
    policy = np.random.rand(num_actions,num_states)
    #print(policy)


    for s in range(num_states):

        #print("state:" +str(s))
        #print( state_action_estimates[:,s])
        a_star = np.argmax( state_action_estimates[:,s])
        #print("opt action:"+str(a_star))

        for a in range(num_actions):

            #print(a)
            if a == a_star:
                policy[a,s] = 1 - epsilon + epsilon/num_actions

            else:
                policy[a,s] = epsilon/num_actions

        #print("the policy for state:"+str(s)+" is:" + str(policy[:,s]))

    states = np.arange(1,num_states+1,1)
    policy = pd.DataFrame(policy,index=actions,columns= states )
    #print(policy)
    return policy

def get_max_action(actions, state,state_action_values):

    a_star = np.argmax(state_action_values[:, state])
    return actions[a_star]


if __name__ == "__main__":

    # define grid world
    grid = Grid.GridWorld(4)
    actions = ["up", "down", "right", "left"]
    transition = tp(grid, actions)
    transition.create_common_transition("Deterministic")
    grid.transition_probs = transition
    reward_env = Reward(grid, actions)
    reward_env.common_reward("sparse")

    # define agent policy
    policy = pd.DataFrame(data=0.25, index=transition.actions, columns=grid.states)

    #define agent
    agent = Agent(grid, actions, policy, reward_env, 1)
    episodes = ve.sample_episode(10, agent, terminal_state=16, steps_per_episode=20)




    counter2 =0
    epsilon = 0.6
    alpha = 0.01
    discount = 0.2
    state_action_values = np.random.rand(len(actions), len(grid.states)) * 5 # random numbers in the range [0,5]

    print(state_action_values)
    print("\n")
    while (counter2 < 1000):
        agent = Agent(grid, actions, policy, reward_env, 1)
        a = agent.next_action()
        counter = 0

        while (agent.current_state != grid.terminal_state and counter < 100):

            s, a, r, s_prime = agent.outcome()
            #print(s,a,r,s_prime)
            #print(state_action_values[:, s_prime -1])
            #print(max(state_action_values[:,s_prime -1]))
            #print(state_action_values[np.argmax(state_action_values[:, s_prime -1]), s_prime - 1])

            state_action_values[actions.index(a), s - 1] = state_action_values[actions.index(a), s - 1] + \
                                                           alpha * (r + discount * max(state_action_values[:,s_prime -1]) -
                                                                    state_action_values[actions.index(a), s - 1])

            counter +=1


        counter2+=1

    print(state_action_values)

    exit(0)


    while(counter2 < 2000):
        agent = Agent(grid, actions, policy, reward_env, 1)
        a = agent.next_action()
        counter =0

        #print("start action: "+ a[0])
        while (agent.current_state != grid.terminal_state and counter < 100):
            #print("Start\n")
            #print(" current state is : " + str(agent.current_state))
            s, a, r, s_prime = agent.outcome(force_action=a)# take a step
            #print (s,a,r,s_prime)
            a_prime = agent.next_action()
            #print("next action:" + a_prime[0])

            state_action_values[actions.index(a),s-1] = state_action_values[actions.index(a),s-1] + \
                                                           alpha*( r + discount*state_action_values[actions.index(a_prime),s_prime-1] - state_action_values[actions.index(a),s-1] )


            policy = epsilon_greedy_policy(state_action_values,epsilon,actions)
            agent.policy = policy
            a = a_prime

            counter +=1

        counter2+=1
    print(policy)
    exit(0)