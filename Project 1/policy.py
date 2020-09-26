import numpy as np

from . import constants


class ActionPolicies:
    def __init__(self):
        self.greedy_epsilon__EPSILON_DEF = float(0.5)

    def greedy_epsilon(self, q_table: np.ndarray,
                       state_dealer: int, state_player: int,
                       epsilon: float = None):
        if epsilon is None:
            epsilon = self.greedy_epsilon__EPSILON_DEF

        _rand_val = np.random.rand()
        # (1-\epsilon) choose the action of max score in the Q-table
        if 0 <= _rand_val < (1 - epsilon):
            # state values <-> array index
            action = np.argmax(q_table[state_dealer - 1, state_player - 1, :])
        # (\epsilon) random choice of action
        elif (1 - epsilon) <= _rand_val <= 1:
            action = np.random.choice(constants.ACTIONS)
        # error
        else:
            raise RuntimeError("Param Value Error: "
                               "\\epsilon @greedy_epsilon policy")

        return action
