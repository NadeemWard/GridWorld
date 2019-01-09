import numpy as np
import pandas as pd


class GridWorld:


    def __init__(self, size, transition_probs=None):

        '''
          - size: size of the square gridworld (just length of one side of the rectangle, not area)
          - transition_probs: a function describing the distributions associated to transitions from state to state P(s_prime| s,a)
        '''
        if (size <= 0):
            print("can't create a gridworld of this size")
            return

        self.size = size
        self.states = np.arange(1, self.size ** 2 + 1, 1)
        self.states = pd.Index(self.states, name="states")
        self.transition_probs = transition_probs
        self.terminal_state = self.states[-1] # assume the terminal state is the bottom right of the grid

    def convert_state(self, state, dim):

        if dim == "1D":

            x_pos = state[0] + 1
            y_pos = state[1] + 1

            state = (y_pos - 1) * self.size + x_pos

        elif dim == "2D":

            if state not in self.states:
                print("invalid state for this gridworld")
                return

            state = max(state - 1, 0)
            y_pos = int(state / self.size)
            y_pos = max(y_pos, 0)

            x_pos = state % self.size
            x_pos = max(x_pos, 0)
            state = (x_pos, y_pos)

        else:
            print("invalid dimension input")
            return

        return state

    def get_next_state(self,state,action):

        '''
        Return the state you end up in starting at state and performing action assuming deterministic env.
        :param state: start state
        :param action: the action performed
        :return: the state you end up in
        '''

        x_pos, y_pos = self.convert_state(state,"2D")

        if (action == 'right'):
            if (not (
                x_pos == self.size - 1)): x_pos += 1  # if we are at the right end of the board state don't increment

        elif (action == 'left'):
            if (not (x_pos == 0)): x_pos -= 1  # if we are at the left end of the board state don't increment

        elif (action == 'up'):
            if (not (y_pos == 0)): y_pos -= 1  # if we are at the upper end of the board state don't increment

        elif (action == 'down'):
            if (not (y_pos == self.size - 1)): y_pos += 1  # if we are at the lower end of the board state don't increment

        return self.convert_state ( (x_pos,y_pos), "1D" )

    def move(self,s,a):

        '''
        Return the state you end up in while taking under consideration the transition probabilities
        :param s: current state
        :param a: action you are taking
        :return: the state you end up in.
        '''
        next_state_probs = []
        for s_prime in self.states:
            next_state_probs.append( self.transition_probs.get(s,a,s_prime) )

        #possible_states = list(self.states.index)

        choice = np.random.choice(self.states, 1, p=next_state_probs) #returns a array of 1 element
        return choice[0]


if __name__ =="__main__":
    x = GridWorld(4)
    print(x.states)