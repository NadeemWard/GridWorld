import Grid
from Transitions import Transitions_Probs as tp
from Agent import Agent
from Rewards import Reward
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_episode(number, agent):
    '''
      Function for getting the episode samples from a given agent
      number: number of samples that we want
      agent: the agent that will generate the samples

      return: list of episodes with each episode consisting of a list of tuples
              (s1,a1,r2,s2) defining the outcomes of the agent.

    '''

    if (number <= 0):
        print("Need to specify a positive number of episodes")
        return

    else:

        episodes = []
        for i in range(number):

            episode = []
            agent = agent.agent_copy()
            while (agent.current_state != agent.gridworld.states[-1]): # assumes termination in the bottom right of grid
                outcome = agent.outcome()
                episode.append(outcome)

            episodes.append(episode)

    return episodes


def value_approximation(grid, policy, reward_env, discount=1, epsilon=10 ** -8):
    '''
        Here, we assume we are using the DP value estimation algorithm as an approximation method for V(s) given policy pi.
      :param policy: The policy to follow pi
      :param reward: the reward function defined by this environment
      :param discount: the discount factor used
      :param epsilon: the stopping criteria, when subsequent iterations lead to a difference less than epsilon we stop
      :return: the approximated state-value estimate V(s)
    '''

    actions = policy.index
    terminal_state = grid.terminal_state # refers to the last state (bottom,right) of a square gridworld
    state_values = np.zeros(terminal_state)  # the V(s) array initialized to zero

    while (True):
        delta = 0
        for s in grid.states:

            if (s == terminal_state): continue  # the terminal state so we don't compute it's state-value

            v = state_values[s - 1]

            sum = 0
            for a in actions:
                policy_proba = policy[s][a]
                # s_prime = grid.move( s, a)
                # r = reward_env.get_reward(s,s_prime,a)# get the reward Rss'a
                # sum += policy_proba * (r + discount * state_values[s_prime - 1])

                for s_prime in grid.states:
                    transition_prob = grid.transition_probs.get(s,a,s_prime)
                    r = reward_env.get_reward(s, s_prime, a)  # get the reward Rss'a
                    sum += policy_proba * ( transition_prob * (r + discount * state_values[s_prime - 1]) )

            state_values[s - 1] = sum

            delta = max(delta,
                        abs(v - state_values[s - 1]))  # just keeping track of the biggest difference between all states

        if (delta < epsilon): return state_values


def get_return(episode, discount=1):
    sum = 0
    index = 0

    for step in episode:
        state, action, reward, next_state = step
        sum += (discount ** index) * reward
        index += 1

    return sum


def monte_carlo(initial_estimates, episodes, state_values,discount =1):

    estimate_Vs = np.copy(initial_estimates)
    avg_abs_differences = []  # a list of results

    returns = [[] for _ in range(len(initial_estimates))]

    difference = abs(state_values - estimate_Vs)
    avg_abs_differences.append(np.average(difference))

    for episode in episodes:

        visited_states = []
        index = 0

        for step in episode:

            state, action, reward, next_state = step

            if (state not in visited_states):
                visited_states.append(state)
                r = get_return(episode[index:], discount=discount)
                returns[state - 1].append(r)
                estimate_Vs[state - 1] = sum(returns[state - 1]) / len(returns[state - 1])

            index+=1

        difference = abs(state_values - estimate_Vs)
        avg_abs_differences.append(np.average(difference))

    return estimate_Vs, avg_abs_differences


def n_step_return(n, episode, state_value_estimate, discount=1):
    '''
      function for returning the target n steps into the future
    '''
    td_steps = n
    end_of_episode = False  # variable to indicate whether the looking ahead will
    # lead to the end of the episode
    sum = 0

    len_episode = len(episode)
    if len_episode <= n:  # length of episode is less than n.
        td_steps = len_episode  # just stop at the end of the episode
        end_of_episode = True

    for index in range(td_steps):
        state, action, reward, next_state = episode[index]
        sum += (discount ** index) * reward

    if not end_of_episode:
        state, action, reward, next_state = episode[td_steps]
        sum += (discount ** (td_steps)) * state_value_estimate[state - 1]

    return sum


def n_step_td(initial_estimates, n, episodes, state_values, discount=1, learning_rate=0.2):

    state_value_estimate = np.copy(initial_estimates)  # the V(s) array initialized to zero
    avg_abs_differences = []  # a list of results

    difference = abs(state_values - state_value_estimate)
    avg_abs_differences.append(np.average(difference))

    # loop through episodes
    for episode in episodes:
        index = 0  # which episode we are at

        for step in episode:
            state, action, reward, next_state = step

            state_value_estimate[state - 1] = state_value_estimate[state - 1] + learning_rate * (
                n_step_return(n, episode[index:], state_value_estimate, discount=discount) - state_value_estimate[
                    state - 1])

            index += 1

        difference = abs(np.array(state_values) - np.array(state_value_estimate))
        avg_abs_differences.append(np.average(difference))

    return state_value_estimate, avg_abs_differences


def td_lambda(initial_estimates, grid, _lambda, episodes, state_values, discount=1, learning_rate=0.2):


    gridworld_size = len(grid.states)
    state_value_estimate = np.copy(initial_estimates)
    e = np.zeros(len(state_value_estimate))  # the elgibility trace
    avg_abs_differences = []

    for episode in episodes:

        for step in episode:

            state, action, reward, next_state = step

            delta = reward + discount * state_value_estimate[next_state - 1] - state_value_estimate[state - 1]
            e[state - 1] = e[state - 1] + 1

            for s in range(gridworld_size):
                state_value_estimate[s] = state_value_estimate[s] + learning_rate * delta * e[s]
                e[s] = discount * _lambda * e[s]

        difference = abs(np.array(state_values) - np.array(state_value_estimate))
        avg_abs_differences.append(np.average(difference))

    return state_value_estimate, avg_abs_differences

if __name__ == "__main__":

    # Define the world
    actions =["up","down","right","left"]
    grid = Grid.GridWorld(4)

    transitions = tp(grid,actions)

    grid.transition_probs = transitions
    transitions.create_common_transition("Deterministic")


    sparse_reward = Reward(grid,actions)
    sparse_reward.common_reward("sparse")

    # Define the Agent
    small_uniform_policy = pd.DataFrame(data=0.25, index=transitions.actions, columns=grid.states)
    agent = Agent(grid,actions,small_uniform_policy,sparse_reward,1)

    episodes = sample_episode(5000,agent)
    #print(episodes)

    # get the true state values
    sparse_uniform_Vs = value_approximation(grid,small_uniform_policy,sparse_reward,0.2)

    # initial estimates
    initial_states = np.random.uniform(-2,2,size=len(grid.states))

    #monte carlo
    vals, MC_sparse_uniform_diffs = monte_carlo(initial_states, grid,episodes, sparse_uniform_Vs,
                                                discount=0.2)

    # n step td
    estimates, avg_abs_diffs = n_step_td(initial_states, 1, episodes, sparse_uniform_Vs,
                                           discount=0.2, learning_rate=0.001)

    #td lambda
    estimate, aads_a = td_lambda(initial_states,grid, 0.5, episodes, sparse_uniform_Vs, discount=0.2,
                               learning_rate=0.001)

    estimate, aads_b = td_lambda(initial_states, grid, 1, episodes, sparse_uniform_Vs, discount=0.2,
                                 learning_rate=0.001)

    estimate, aads_c = td_lambda(initial_states, grid, 0, episodes, sparse_uniform_Vs, discount=0.2,
                                 learning_rate=0.001)
    # plot

    plt.plot(avg_abs_diffs)
    #plt.plot(aads_a)
    #plt.plot(aads_b)
    #plt.plot(aads_c)
    #plt.plot(MC_sparse_uniform_diffs[:200])
    #plt.legend(['td','MC'])
    plt.ylabel("Average absolute difference")
    plt.xlabel("# of episodes")
    plt.show()