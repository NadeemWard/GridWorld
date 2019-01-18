import numpy as np
import math


class GridWorld:

    def __init__(self, height, width, transition_probs=None, reward_env=None, terminal_state = None):
        '''
        Class to create a Grid World
        :param height: the height of the grid
        :param width: the width of the grid
        :param transition_probs: transition probabilities associated to grid, default to none.
        :param terminal_state: default to none, if you want specify it you can.
        '''

        if (height <= 0 or width <= 0):
            print("can't create a gridworld of this size")
            return

        # define the size of the grid and the number of states
        self.height = height
        self.width = width
        self.states = np.arange(1, (self.height * self.width) + 1 , 1)


        self.transition_probs = transition_probs

        if(terminal_state == None):
            self.terminal_state = height*width # default to the last state
        elif(terminal_state in self.states):
            self.terminal_state = terminal_state
        else:
            print("Invalid terminal state")
            return

        self.reward_env = reward_env

    def convert_state(self, state, dim):
        '''
        convert from 1D representation of state (1,2,...) to 2D representation (x,y). Note that the 2D representation
        follows the normal way we define matrix positions with the top left corner (1,1) and bottom right (height,width)

        :param state: either 1D number s1 or tuple (x,y)
        :param dim: the dimension to convert to. either "1D" or "2D"
        :return: the converted representation.
        '''

        if dim == "1D":

            x_pos = state[0]
            y_pos = state[1]

            state = (y_pos) + (x_pos -1)* self.width

        elif dim == "2D":

            if state not in self.states:
                print("invalid state for this gridworld")
                return

            # convert x pos
            x_pos = math.ceil(state/self.width)
            # convert y pos
            y_pos = state % self.width
            if(y_pos == 0): y_pos = self.width

            state = (x_pos, y_pos)

        else:
            print("invalid dimension input")
            return

        return state

    def get_next_state(self,state,action):

        '''
        Return the state you end up in starting at state and performing action assuming deterministic env.

        :param state: start state
        :param action: the action performed. A string either "right","left","up","down"
        :return: the state you end up in
        '''

        x_pos, y_pos = self.convert_state(state,"2D")

        if (action == 'right'):
            if (not (
                y_pos == self.width )): y_pos += 1  # if we are at the right end of the board state don't increment

        elif (action == 'left'):
            if (not (y_pos == 1)): y_pos -= 1  # if we are at the left end of the board state don't increment

        elif (action == 'up'):
            if (not (x_pos == 1)): x_pos -= 1  # if we are at the upper end of the board state don't increment

        elif (action == 'down'):
            if (not (x_pos == self.height ) ): x_pos += 1  # if we are at the lower end of the board state don't increment

        return self.convert_state ( (x_pos,y_pos), "1D" )

    def print_grid(self):

        output = "|"
        for i in self.states:

            if(i % self.width == 1 and i != 1): output+="|\n|"

            output += str(i) +"\t"

        output+="|"

        print(output)

    def move(self,s,a):

        '''
        Return the state you end up in while taking under consideration the transition probabilities
        :param s: current state
        :param a: action you are taking
        :return: the state you end up in.
        '''

        if(self.transition_probs == None):
            print("Transition probabilities not yet define")
            return

        next_state_probs = []
        for s_prime in self.states:
            next_state_probs.append( self.transition_probs.get(s,a,s_prime) )

        #possible_states = list(self.states.index)

        choice = np.random.choice(self.states, 1, p=next_state_probs) #returns a array of 1 element
        return choice[0]


if __name__ =="__main__":

    grid = GridWorld(10,10)
    grid.print_grid()

    from Transitions import Transitions_Probs

    actions = ["up", "down", "right", "left"]
    x = Transitions_Probs(grid,actions)
    x.create_common_transition("Deterministic") #("Bernoulli",0.7)) # "Deterministic"









