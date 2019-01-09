import numpy as np

class Reward:

    def __init__(self,grid,actions):

        self.grid = grid
        self.actions = actions
        self.expected_rewards = np.zeros((len(grid.states),len(actions),len(grid.states)))

    def convert(self,action):
        '''
        Convert an action from a string to its index in the matrix
        :param action: the string action
        :return: the index for this action in the matrix
        '''

        return self.actions.index(action)

    def common_reward(self,common_type):

        if(common_type == "sparse"):

            for s in self.grid.states:
                for a in self.actions:
                    self.expected_rewards[s-1][self.convert(a)][self.grid.terminal_state-1] = 1

        if(common_type =="dense"):
            # all states that are multiples of 2 are given a reward of 1
            # to give half of the states a positive reward, therefore dense

            for s in self.grid.states:
                for a in self.actions:
                    for s_prime in self.grid.states:
                        if s_prime % 2 == 0:
                            self.expected_rewards[s - 1][self.convert(a)][s_prime - 1] = 1



    def get_reward(self,s,s_prime,a):
        '''
        Assume the rewards are deterministic, just return the expected value. Not a sample from a distr
        :param s: start state
        :param s_prime: end state
        :param a: action
        :return: reward
        '''

        return self.expected_rewards[s-1][self.convert(a)][s_prime-1]