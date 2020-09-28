import numpy as np
import typing

import constants


class UpdateQTable:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def q_function(self, q_table: np.ndarray,
                   state_crt, action: int, reward: float,
                   state_next) -> float:
        new_val = 0.

        # 1-st item
        new_val += (1 - self.learning_rate) * \
                   q_table[state_crt[0] - 1, state_crt[1] - 1, action]
        if constants.DEBUG_DEBUG:
            print("\t", new_val, end=", ")
        # 2-nd item
        new_val += self.learning_rate * reward
        if constants.DEBUG_DEBUG:
            print(new_val, end=", ")
        # 3-rd item
        if constants.judge_state_is_terminate(state_next):
            pass
        else:
            new_val += self.learning_rate * self.discount_factor * \
                       np.max(q_table[state_next[0] - 1, state_next[1] - 1, :])
        if constants.DEBUG_DEBUG:
            print(new_val)

        return new_val


if "__main__" == __name__:
    update_obj = UpdateQTable()  # learning_rate=0.1, discount_factor=0.5
    test_q_table = np.abs(np.random.randn(*constants.STATE_SPACE_SHAPE))
    test_new_val = update_obj.q_function(
        q_table=test_q_table, state_crt=(5, 7), action=0,
        reward=0, state_next=(5, 9))
    print(test_new_val)
