import Grid
from Transitions import Transitions_Probs as tp
from Agent import Agent
from Rewards import Reward
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Value_estimation as ve
import sys


def avg_diff_vs_learning_rate(initial_states, steps, learning_rates, episodes, state_values):

    '''

    :param initial_states:
    :param steps:
    :param learning_rates:
    :param episodes:
    :param state_values:
    :return: a dictionary of n-step-method: [aad with learning rate 1, aad with learning rate 2 ,... ]
    '''

    method_aads = {}
    for n in steps:
        all_aads = []
        for alpha in learning_rates:

            estimates, avg_abs_diffs = ve.n_step_td(initial_states, n, episodes, state_values,
                                                    discount=0.2, learning_rate=alpha)
            all_aads.append(avg_abs_diffs[-1])

        method_aads["{0}-step-TD".format(n)] = all_aads

    return method_aads


def plot_aads(aads, learning_rates, title):
    fig, ax = plt.subplots()
    a = [1, 2, 3, 4, 5, 6, 7, 8]

    for name, y in aads.items():
        ax.plot(a, y)

    ax.set_xticks(a)
    ax.set_xticklabels(learning_rates)
    plt.legend(list(aads.keys()))
    plt.ylabel("Average absolute difference")
    plt.xlabel("Learning rates")
    plt.title(title)
    plt.show()


def plot_aads_for_methods(method_aads_dict, title, number_episodes):
    for method, aads in method_aads_dict.items():
        plt.plot(aads[:number_episodes])

    plt.legend(list(method_aads_dict.keys()))
    plt.ylabel("Average absolute difference")
    plt.xlabel("# of episodes")
    plt.title(title)
    plt.show()


if __name__ =="__main__":

    # example of running a TD method.
    # Here we choose to study the small gridworld with sparse reward and a stochastic transition probability using
    # and agent with a uniform policy.

    initial_states = np.random.uniform(-2, 2, size=16)
    episodes = np.load("episodes/small_Bernoulli_uniform_sparse_grid.npy")
    state_values = np.load("state_values/small_Bernoulli_uniform_sparse_discount_of_0.2_state_value.npy")

    print(episodes[0])
    print("\n")
    print(episodes[1])
    print("\n")
    print(initial_states)
    print("\n")
    print(state_values)

    estimates, avg_abs_diffs = ve.n_step_td(initial_states, 10, episodes, state_values,
                                            discount=0.2, learning_rate=0.001)
    plt.plot(avg_abs_diffs[:1500])
    plt.ylabel("Average absolute difference")
    plt.xlabel("# of episodes")
    plt.show()

    vals, avg_abs_diffs = ve.monte_carlo(initial_states, episodes, state_values,
                                   discount=0.2)

    print(avg_abs_diffs[0])
    print("\n")
    print(avg_abs_diffs[1])
    print("\n")
    print(avg_abs_diffs[2])


    plt.plot(avg_abs_diffs[:500])
    plt.ylabel("Average absolute difference")
    plt.xlabel("# of episodes")
    plt.show()


    exit(0)

    # Define the world
    actions = ["up", "down", "right", "left"]
    size = ["small", "big"]
    transition_type = ["Deterministic", "Bernoulli"]
    policy_type = ["uniform", "intuitive"]
    reward_type = ["sparse", "dense"]

    # finding the best learning rate
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.5, 1]
    n_step_methods = [1, 2, 6, 10, 50]
    gridworld_size = 4
    discount = 0.2

    best_learning_rates = [0.001, 0.001, 0.1, 0.01, 0.001, 0.001, 1, 0.2]
    method_aads = {}
    index = 0

    for s in size:
        if s == "small":
            initial_states = np.random.uniform(-2, 2, size=16)
        else:
            initial_states = np.random.uniform(-2, 2, size=100)

        for p in policy_type:
            for r in reward_type:

                episodes = np.load("episodes/{0}_Bernoulli_{1}_{2}_grid.npy".format(s,p,r) ) #Always bernoulli transitions
                state_values = np.load("state_values/{0}_Bernoulli_{1}_{2}_discount_of_{3}_state_value.npy".format(s,p,r,discount) ) #always bernoulli transitions


                for n in n_step_methods:
                    estimates, avg_abs_diffs = ve.n_step_td(initial_states, n, episodes, state_values,
                                                        discount=0.2, learning_rate=best_learning_rates[index])
                    method_aads["{0}_step_TD".format(n)] = avg_abs_diffs

                # add monte Carlo
                vals, MC_aads = ve.monte_carlo(initial_states, episodes, state_values,
                                                            discount=0.2)
                method_aads["Monte Carlo"] = MC_aads
                print("here")
                title = "Performance of n_step_TD methods on {3} grid with alpha={0}, {1} reward environment with {2} policy".format(best_learning_rates[index],r,p,s)
                plot_aads_for_methods(method_aads, title, 2500)

                index +=1

                # method_aads = avg_diff_vs_learning_rate(initial_states,n_step_methods,learning_rates,episodes,state_values)
                # title = "{0} GridWorld On-line n-step TD on the {1} policy with {2} rewards ".format(s,p,r)
                # plot_aads(method_aads, learning_rates, title)





    ####################################################################################################################

    #
    # grid = Grid.GridWorld(4)
    #
    # transitions = tp(grid, actions)
    # transitions.create_common_transition("Deterministic")
    # grid.transition_probs = transitions
    #
    # # # load data
    # # data = np.load("episodes/big_Bernoulli_uniform_dense_grid.npy")
    # #
    # sparse_reward = Reward(grid,actions)
    # sparse_reward.common_reward("sparse")
    #
    # small_uniform_policy = pd.DataFrame(data=0.25, index=transitions.actions, columns=grid.states)
    # small_intuitive_policy = pd.DataFrame(data=0, index=actions, columns=grid.states)
    # for columns in small_intuitive_policy:
    #     small_intuitive_policy.loc[small_intuitive_policy.index[[2]], columns] = 0.8  # right
    #     small_intuitive_policy.loc[small_intuitive_policy.index[[1]], columns] = 0.2  # down
    #
    #
    # print(small_intuitive_policy)
    # #print(grid.transition_probs)
    # # get the true state values
    # sparse_uniform_Vs = ve.value_approximation(grid, small_intuitive_policy, sparse_reward, 0.2)
    # #
    # # # initial estimates
    # # initial_states = np.random.uniform(-2, 2, size=len(grid.states))
    # #
    # print(sparse_uniform_Vs)
    # print("-----------------------")
    # data = np.load("state_values/small_Deterministic_intuitive_sparse_discount_of_0.2_state_value.npy")
    # print(data)
    # print(initial_states)
    #
    # # difference = abs(sparse_uniform_Vs - initial_states)
    # print(np.average(difference))
    #
    # estimates, avg_abs_diffs1 = ve.n_step_td(initial_states, 1, episodes, sparse_uniform_Vs,
    #                                      discount=0.2, learning_rate=0.01)
    #
    # estimates, avg_abs_diffs2 = ve.n_step_td(initial_states, 1, episodes, sparse_uniform_Vs,
    #                                         discount=0.2, learning_rate=0.1)
    #
    # estimates, avg_abs_diffs3 = ve.n_step_td(initial_states, 1, episodes, sparse_uniform_Vs,
    #                                         discount=0.2, learning_rate=0.001)
    #
    # vals, MC_sparse_uniform_diffs = ve.monte_carlo(initial_states, grid, episodes, sparse_uniform_Vs,
    #                                             discount=0.01)
    #
    # plt.plot(avg_abs_diffs1[:2000])
    # plt.plot(avg_abs_diffs2[:2000])
    # plt.plot(avg_abs_diffs3[:2000])
    # plt.plot(MC_sparse_uniform_diffs[:2000])
    # plt.legend(['1-step learning rate 0.01','1-step learning rate 0.1','1-step learning rate 0.001',"MC"])
    # plt.ylabel("Average absolute difference")
    # plt.xlabel("# of episodes")
    # plt.show()