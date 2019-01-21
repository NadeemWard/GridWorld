import numpy as np
import sys

class Agent:

    def __init__(self, gridworld, actions, policy, start_state=1):

        self.gridworld = gridworld
        self.actions = actions
        self.policy = policy

        if (start_state not in gridworld.states):
            print("invalid start state for the gridworld")
            return
        self.current_state = start_state  # starting state
        self.start_state = start_state

    def agent_copy(self):
        return Agent(self.gridworld, self.actions, self.policy, self.start_state) # the agent will start at start_state

    def get_state(self):
        return self.current_state

    def next_action(self):
        '''
        get the next action from the current state given the policy
        :param policy: the policy to follow
        :param state: the current state
        :return: the move the agent will take following the policy
        '''
        #possible_actions = actions
        return np.random.choice(self.actions, 1, p=self.policy[self.current_state -1])

    def outcome(self, force_action=None):
        '''
        Function for getting data from an agent
        :param force_action: if you want to force an action to be taken (for action value estimation)
        :return: the complete tuple (state, action, reward, next_state)
        '''
        action = self.next_action()

        if (force_action!= None):
            action = force_action

        previous_state = self.current_state
        self.current_state = self.gridworld.move(self.current_state,action)

        return (
        previous_state, action[0], self.gridworld.reward_env.get_reward(previous_state, self.current_state, action), self.current_state)

    def sample_episode(self, number_of_episodes, terminal_state = None, steps_per_episode = None):
        '''
          Function for getting the sample episodes from an agent following a given policy. We would like agent to terminate
          either after they reach a specified state or after a certain number of steps.

          number_of_episodes: number of episodes that we want
          terminal_state = specify the state you would like to terminate an episode at
          steps_per_episode = specify the number of steps you would like the agent to take before terminating the episode.

          return: list of episodes with each episode consisting of a list of tuples
                  (s1,a1,r2,s2) defining the outcomes of the agent.

        '''

        if (number_of_episodes <= 0):
            print("Need to specify a positive number of episodes")
            return

        if terminal_state == None and steps_per_episode == None:
            # default termination of agent if none specified
            terminal_state = self.gridworld.states[-1]
            steps_per_episode = sys.maxsize

        elif terminal_state == None and steps_per_episode <= 0:
            print("Incorrect value for the number of steps per episode")
            return

        elif terminal_state not in self.gridworld.states and steps_per_episode == None:
            print("Incorrect terminal state specified.")
            return

        elif terminal_state not in self.gridworld.states and steps_per_episode <= 0:
            print("Invalid Input")
            return

        if(steps_per_episode == None):
            steps_per_episode = sys.maxsize

        episodes = []
        for i in range(number_of_episodes):

            episode = []
            agent = self.agent_copy()
            number_steps = 0
            while (
                    agent.current_state != terminal_state and number_steps < steps_per_episode):  # assumes termination in the bottom right of grid

                outcome = agent.outcome()
                episode.append(outcome)

                number_steps += 1

            episodes.append(episode)

        return episodes

if __name__ =="__main__":

    height = 4  # square gridworld
    width = 4
    import Grid
    grid = Grid.GridWorld(height, width)
    grid.print_grid()

    actions = ["up", "down", "right", "left"]

    import Transitions


    x = Transitions.Transitions_Probs(grid, actions)
    x.create_common_transition("Deterministic")  # ("Bernoulli",0.7)) # "Deterministic"

    import Rewards
    sparse_reward = Rewards.Reward(grid, actions)
    sparse_reward.common_reward("sparse")

    policy = np.ones((len(grid.states), len(actions) )) * 0.25 # uniform policy

    # go right 80% of time and down 20%
    policy = np.zeros((len(grid.states), len(actions)))
    for state in policy:
        #print(i)
        state[actions.index("right")] = 0.8
        state[actions.index("down")] = 0.2

    print(policy)

    agent = Agent(grid, actions, policy)
    print(agent.sample_episode(10)) # get 10 sample episodes

    exit(0)


    for s in range(10):
        print(agent.outcome() )