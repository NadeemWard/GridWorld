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

        for s in terminal_states:

            for a in actions:

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

    x.add_terminal_states([1,5])

    print("\n")
    print("HERERERERE")
    print(x.t_probs)

    for i in range(5):
        print(x.t_probs[0][i])
        print(x.t_probs[4][i])
    # print(x.get(6,"up",7))
    # print(x.get(6,"right",7))
    # print(x.get(6,"right",6))

    # for s in grid.states:
    #     for a in actions:
    #         print("-------------------")
    #         print(s)
    #         print(a)
    #         result = grid.move(s,a)
    #         print(result)
    #         if grid.get_next_state(s,a) != result: print("HERE")
            #print(grid.move(s,a))
    #print(grid.move(6,"right"))

    ############################################

            # if(s_others != s_true):
            #
            #     self.t_probs[s-1][self.convert(a)][s_others -1] += random_prob
            # else:
            #     self.t_probs[s - 1][self.convert(a)][s_others - 1] += random_prob
