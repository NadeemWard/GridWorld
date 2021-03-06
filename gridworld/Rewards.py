import numpy as np

class Reward:

    def __init__(self,grid,actions):

        self.grid = grid
        self.actions = actions
        self.expected_rewards = np.zeros((len(grid.states),len(actions),len(grid.states))) # which represents reward starting at state s,
                                                                                           # taking action a and landing in state s_t+1
        grid.reward_env = self

    def convert(self,action):
        '''
        Convert an action from a string to its index in the matrix
        :param action: the string action
        :return: the index for this action in the matrix
        '''

        return self.actions.index(action)

    def common_reward(self,common_type):

        if(isinstance(common_type,dict)):

            for state, reward in common_type.items():

                for s in self.grid.states:
                    for a in self.actions:
                        self.expected_rewards[s - 1][self.convert(a)][state - 1] = reward

        if(common_type == "sparse"):

            for s in self.grid.states:
                for a in self.actions:
                    self.expected_rewards[s-1][self.convert(a)][self.grid.terminal_state-1] = 1

        elif(common_type =="dense"):
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

    def add_terminal_states(self,states):

        # Terminal state means that the reward is 0 for all action starting from that state R(s_terminal , a, s) = 0
        for state in states:
            for s in self.grid.states:
                for a in self.actions:
                    self.expected_rewards[state - 1][ self.convert(a) ][s - 1] = 0