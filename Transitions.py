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

        if(common_type[0] == "Bernoulli"):

            self.t_probs = np.zeros(
                (len(self.states), len(self.actions), len(self.states)))  # This considers all entries to be 0

            sucess = common_type[1]
            failure = 1- sucess

            for s in self.states:

                for a in self.actions:

                    s_true = grid.get_next_state(s,a) # the true state youll land in if you are in state s and follow action a

                    self.t_probs[s-1][self.convert(a)][s_true-1] = sucess # give the true state the probability of success.
                    if(s_true != s):
                        self.t_probs[s - 1][self.convert(a)][s - 1] = failure # give the proba of failure to staying were you are.
                    else:
                        self.t_probs[s - 1][self.convert(a)][s - 1] += failure

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
    height = 4 # square gridworld
    width = 4
    import Grid
    grid = Grid.GridWorld(height, width)
    grid.print_grid()

    x = Transitions_Probs(grid,actions)
    x.create_common_transition("Deterministic") #("Bernoulli",0.7)) # "Deterministic"

    # print(x.t_probs)
    # print(x.get(6,"up",7))
    # print(x.get(6,"right",7))
    # print(x.get(6,"right",6))

    for s in grid.states:
        for a in actions:
            print("-------------------")
            print(s)
            print(a)
            print(grid.move(s,a))
    #print(grid.move(6,"right"))
