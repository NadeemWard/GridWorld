import Grid
import numpy as np
import pandas as pd
import Transitions
import Rewards
#from Value_estimation import sparse_reward_env

class Agent:

    def __init__(self, gridworld, actions, policy, reward_env, start_state):

        self.gridworld = gridworld
        self.actions = actions
        self.policy = policy

        if (start_state not in gridworld.states):
            print("invalid start state for the gridworld")
            return
        self.current_state = start_state  # starting state
        self.start_state = start_state
        self.reward_env = reward_env

    def agent_copy(self):
        return Agent(self.gridworld, self.actions, self.policy, self.reward_env, self.start_state)

    def get_state(self):
        return self.current_state

    def next_action(self):
        '''
        get the next action from the current state given the policy
        :param policy: the policy to follow
        :param state: the current state
        :return: the move the agent will take following the policy
        '''
        possible_actions = list(self.policy.index)
        return np.random.choice(possible_actions, 1, p=self.policy[self.current_state])

    # def reward_function(self, s, s_prime, a):
    #
    #     '''
    #     reward for moving form state s to state s_prime using action a
    #     '''
    #
    #     return self.reward_env[s_prime - 1]

    def outcome(self):

        action = self.next_action()
        previous_state = self.current_state
        self.current_state = self.gridworld.move(self.current_state,action)

        return (
        previous_state, action[0], self.reward_env.get_reward(previous_state, self.current_state, action), self.current_state)


if __name__ =="__main__":

    actions = ["up", "down", "right", "left"]
    size = 4  # square gridworld

    grid = Grid.GridWorld(size)
    print(grid.terminal_state)

    x = Transitions.Transitions_Probs(grid, actions)

    grid.transition_probs = x

    x.create_common_transition(("Bernoulli", 0.7))  # "Deterministic"

    # for s in grid.states:
    #     for a in actions:
    #         print("-------------------")
    #         print(s)
    #         print(a)
    #         print(grid.move(s,a))


    sparse_reward = Rewards.Reward(grid,actions)
    sparse_reward.common_reward("sparse")
    print(sparse_reward.expected_rewards)

    small_uniform_policy = pd.DataFrame(data=0.25, index=x.actions, columns=grid.states)
    print(small_uniform_policy)
    agent = Agent(grid,actions,small_uniform_policy,sparse_reward,1)

    episode = []

    while(agent.current_state != agent.gridworld.states[-1]):

        outcome = agent.outcome()


        episode.append(outcome)

    print(episode)