import numpy as np
import typing

from . import constants


class UpdateQTable:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def q_function(self, q_table: np.ndarray,
                   state_crt, action: int, reward: float,
                   state_next) -> None:
        new_val = 0.

        # 1-st item
        new_val += (1 - self.learning_rate) * q_table[state_crt[0], state_crt[1], action]
        # 2-nd item
        new_val += self.learning_rate * reward
        # 3-rd item
        if "TERMINAL" == state_next[0] and "TERMINAL" == state_next[1]:
            pass
        else:
            new_val += self.learning_rate * self.discount_factor * np.amax(q_table[state_next[0], state_next[1], :])
        q_table[state_crt[0], state_crt[1], action] = new_val
