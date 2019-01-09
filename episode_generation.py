import Grid
from Transitions import Transitions_Probs as tp
from Agent import Agent
from Rewards import Reward
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Value_estimation as ve

if __name__ =="__main__":


    # Define the grid worlds
    actions = ["up", "down", "right", "left"]

    size =["small","big"]
    transition_type = ["Deterministic","Bernoulli"]
    policy_type = ["uniform","intuitive"]
    reward_type =["sparse","dense"]


    #data = np.load("episodes/big_Bernoulli_uniform_dense_grid.npy")

    grids = {}
    for s in size:

        if s =="small":
            grid = Grid.GridWorld(4)

        else:
            grid = Grid.GridWorld(10)

        for t in transition_type:

            transition = tp(grid, actions)
            if t == "Deterministic":
                transition.create_common_transition(t)

            else:
                transition.create_common_transition((t,0.8))

            grid.transition_probs = transition

            for p in policy_type:

                if p == "uniform":
                    policy = pd.DataFrame(data=0.25, index=transition.actions, columns=grid.states)

                else:
                    policy = pd.DataFrame(data=0, index=transition.actions, columns=grid.states)
                    for columns in policy:
                        policy.loc[policy.index[[2]], columns] = 0.8  # right
                        policy.loc[policy.index[[1]], columns] = 0.2  # down

                for r in reward_type:

                    reward_env = Reward(grid,actions)
                    reward_env.common_reward(r)

                    name = "{0}_{1}_{2}_{3}_grid".format(s,t,p,r)

                    grids[name] =(grid,reward_env,policy)



    for name, items in grids.items():
        grid, reward_env, policy = items
        print(name)
        print(policy)
        agent = Agent(grid,actions, policy,reward_env,1)

        if "small" in name:
            episodes = ve.sample_episode(10000, agent)
        else:
            episodes = ve.sample_episode(2000, agent)

        np.save("episodes/{0}".format(name), episodes)




    ################################################################################
       ################# generate the true state values by approximation using dynamic programming
    # state_values = {}
    # discount = 0.2
    #
    # for s in size:
    #
    #     if s =="small":
    #         grid = Grid.GridWorld(4)
    #
    #     else:
    #         grid = Grid.GridWorld(10)
    #
    #     for t in transition_type:
    #
    #         transition = tp(grid, actions)
    #         if t == "Deterministic":
    #             transition.create_common_transition(t)
    #
    #         else:
    #             transition.create_common_transition((t,0.8))
    #
    #         grid.transition_probs = transition
    #
    #         for p in policy_type:
    #
    #             if p == "uniform":
    #                 policy = pd.DataFrame(data=0.25, index=transition.actions, columns=grid.states)
    #
    #             else:
    #                 policy = pd.DataFrame(data=0, index=transition.actions, columns=grid.states)
    #                 for columns in policy:
    #                     policy.loc[policy.index[[2]], columns] = 0.8  # right
    #                     policy.loc[policy.index[[1]], columns] = 0.2  # down
    #
    #             for r in reward_type:
    #
    #                 reward_env = Reward(grid,actions)
    #                 reward_env.common_reward(r)
    #
    #
    #
    #                 name = "{0}_{1}_{2}_{3}_discount_of_{4}_state_value".format(s,t,p,r,discount)
    #                 print(name)
    #                 state_values[name] = ve.value_approximation(grid, policy, reward_env, discount)
    #
    #
    # for name, value in state_values.items():
    #     print("here")
    #     np.save("state_values/{0}".format(name), value)
