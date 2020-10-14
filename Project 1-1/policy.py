import numpy as np
import typing

import constants


class ActionPolicies:
    def __init__(self):
        pass

    @staticmethod
    def greedy_epsilon(q_table: np.ndarray,
                       state: typing.List[int],
                       epsilon: float = 0.5) -> int:
        state_dealer, state_player = state

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

    def greedy_maximum(self, q_table: np.ndarray, state: typing.List[int], **kwargs) -> int:
        return self.greedy_epsilon(q_table=q_table, state=state, epsilon=0.)


if "__main__" == __name__:
    policy_obj = ActionPolicies()
    test_q_table = np.abs(np.random.randn(*constants.STATE_ACTION_SPACE_SHAPE))
    test_action = policy_obj.greedy_epsilon(
        q_table=test_q_table, state=[5, 7], epsilon=0.5)
    print(test_action)
