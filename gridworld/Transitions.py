import numpy as np

class Transitions_Probs:

    def __init__(self,grid, actions, data = None):
        '''
        Defining the environment states and actions. We assume that we transition to the same states set.
        :param actions: the actions of the environment. Have it be a list of strings.
        :param data: the data we want to store, in the matrix of transition probabilities. If none fill to zero
        :param grid: The grid these transition probabilities are acting on.
        '''

        self.states = grid.states
        self.actions = actions
        self.grid = grid
        ''' create a 3D matrix of size |S|x|A|x|S| '''
        self.t_probs = np.zeros((len(self.states),len(actions),len(self.states)) )
        grid.transition_probs = self

        if (data != None):
            # input the data provided.
            self.t_probs = data

    def convert(self,action):
        '''
        Convert an action from a string to its index in the matrix
        :param action: the string action
        :return: the index for this action in the matrix
        '''

        return self.actions.index(action)

    def add_terminal_states(self,terminal_states):
        '''
        This function adds the specified terminal states by making them absorbing states.
        i.e. P(s'|s,a) = 0 if s is terminal at s' != s for all actions a.
            P(s'|s,a) = 1 if s' = s and it is a terminal states. For all actions a.
        Think of it as a markov chain where this state has one outward arrow that loops back to itself with
        proba 1.
        :param terminal_states: the state to be made terminal
        :return: nothing, just updates the transition probabilities.
        '''

        for s in terminal_states:

            for a in self.actions:

                for s_prime in self.grid.states:

                    if (s_prime == s):
                        self.t_probs[s-1][self.convert(a)][s_prime - 1] = 1
                    else:
                        self.t_probs[s - 1][self.convert(a)][s_prime -1] = 0



    def create_common_transition(self,common_type):

        '''

        :param common_type: either string "Deterministic" or tuple ("Bernoulli",probability of success)
        :return: nothing. Just modifies the transition probabilities
        '''


        if(common_type == "Deterministic"):

            self.t_probs = np.zeros((len(self.states), len(self.actions), len(self.states)))  #This considers all entries to be 0

            for s in self.states:

                for a in self.actions:

                    s_true = self.grid.get_next_state(s,a)# the true state youll land in if you are in state s and follow action a

                    self.t_probs[s-1][self.convert(a)][s_true-1] = 1 # give the true state a probability of 1.

        elif(common_type[0] == "Bernoulli"):

            self.t_probs = np.zeros(
                (len(self.states), len(self.actions), len(self.states)))  # This considers all entries to be 0

            sucess = common_type[1]
            failure = 1- sucess

            for s in self.states:

                for a in self.actions:

                    s_true = self.grid.get_next_state(s,a) # the true state youll land in if you are in state s and follow action a

                    self.t_probs[s-1][self.convert(a)][s_true-1] = sucess # give the true state the probability of success.
                    if(s_true != s):
                        self.t_probs[s - 1][self.convert(a)][s - 1] = failure # give the proba of failure to staying were you are.
                    else:
                        self.t_probs[s - 1][self.convert(a)][s - 1] += failure

        elif(common_type[0]== "Random"):

            self.t_probs = np.zeros(
                (len(self.states), len(self.actions), len(self.states)))  # This considers all entries to be 0

            success = common_type[1]
            failure = 1 - success
            random_prob = failure/len(self.actions)

            for s in self.states:

                for a in self.actions:

                    s_true = self.grid.get_next_state(s,a)

                    self.t_probs[s - 1][self.convert(a)][s_true - 1] = success + random_prob

                    for other_a in self.actions:

                        if( other_a != a ):
                            s_others = self.grid.get_next_state(s,other_a)
                            self.t_probs[s - 1][self.convert(a)][s_others - 1] += random_prob

    def get(self,s,a,s_prime):

        '''
        get the transition probability
        :param s:  start in s
        :param a: perform action a
        :param s_prime: end up in s_prime
        :return: the transition probability associated
        '''
        r = self.t_probs[s-1][self.convert(a)][s_prime-1]
        return r



if __name__ =="__main__":

    actions = ["up", "down", "right", "left"]
    height = 5 # square gridworld
    width = 5
    import Grid
    grid = Grid.GridWorld(height, width)
    grid.print_grid()

    x = Transitions_Probs(grid,actions)
    x.create_common_transition(("Random",0.7)) # "Deterministic"

    print(x.t_probs)
    print(x.t_probs[0][0])

    grid.add_terminal_states([1,5])
