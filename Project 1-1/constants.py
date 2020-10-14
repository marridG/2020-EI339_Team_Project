import numpy as np

DECK = range(1, 10 + 1)
ACTIONS = (HIT, STICK) = (0, 1)

DEALER_RANGE = range(1, 10 + 1)  # value of the dealer's first card
PLAYER_RANGE = range(1, 21 + 1)
STATE_ACTION_SPACE_SHAPE = (len(DEALER_RANGE), len(PLAYER_RANGE), len(ACTIONS))
STATE_SPACE_SHAPE = (len(DEALER_RANGE), len(PLAYER_RANGE))
STATE_SPACE = np.array([[dealer, player] for dealer in DEALER_RANGE for player in PLAYER_RANGE])

TERMINAL_STATE = "TERMINAL"
DEALER_STICK_THRES = 16
COLOR_PROBS = {"RED": 1 / 3, "BLACK": 2 / 3}
COLOR_COEFFS = {"RED": -1, "BLACK": 1}

# DEBUG
DEBUG_MODE = False
DEBUG_INFO = DEBUG_MODE and True  # False  #
DEBUG_DEBUG = DEBUG_MODE and True  # False  #


def judge_state_is_terminate(state):
    is_terminate = TERMINAL_STATE == state[0] \
                   and TERMINAL_STATE == state[1]
    return is_terminate
