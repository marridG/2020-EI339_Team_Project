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

# TRAINING SETTINGS
TR_EPISODE = 10000


def judge_state_is_terminate(state):
    is_terminate = TERMINAL_STATE == state[0] \
                   and TERMINAL_STATE == state[1]
    return is_terminate


state_trans_hit_prob = {
    1: {-9: 0.03333333333333333, -8: 0.03333333333333333, -7: 0.03333333333333333, -6: 0.03333333333333333,
        -5: 0.03333333333333333, -4: 0.03333333333333333, -3: 0.03333333333333333, -2: 0.03333333333333333,
        -1: 0.03333333333333333, 0: 0.03333333333333333, 2: 0.06666666666666667, 3: 0.06666666666666667,
        4: 0.06666666666666667, 5: 0.06666666666666667, 6: 0.06666666666666667, 7: 0.06666666666666667,
        8: 0.06666666666666667, 9: 0.06666666666666667, 10: 0.06666666666666667, 11: 0.06666666666666667},
    2: {-8: 0.03333333333333333, -7: 0.03333333333333333, -6: 0.03333333333333333, -5: 0.03333333333333333,
        -4: 0.03333333333333333, -3: 0.03333333333333333, -2: 0.03333333333333333, -1: 0.03333333333333333,
        0: 0.03333333333333333, 1: 0.03333333333333333, 3: 0.06666666666666667, 4: 0.06666666666666667,
        5: 0.06666666666666667, 6: 0.06666666666666667, 7: 0.06666666666666667, 8: 0.06666666666666667,
        9: 0.06666666666666667, 10: 0.06666666666666667, 11: 0.06666666666666667, 12: 0.06666666666666667},
    3: {-7: 0.03333333333333333, -6: 0.03333333333333333, -5: 0.03333333333333333, -4: 0.03333333333333333,
        -3: 0.03333333333333333, -2: 0.03333333333333333, -1: 0.03333333333333333, 0: 0.03333333333333333,
        1: 0.03333333333333333, 2: 0.03333333333333333, 4: 0.06666666666666667, 5: 0.06666666666666667,
        6: 0.06666666666666667, 7: 0.06666666666666667, 8: 0.06666666666666667, 9: 0.06666666666666667,
        10: 0.06666666666666667, 11: 0.06666666666666667, 12: 0.06666666666666667, 13: 0.06666666666666667},
    4: {-6: 0.03333333333333333, -5: 0.03333333333333333, -4: 0.03333333333333333, -3: 0.03333333333333333,
        -2: 0.03333333333333333, -1: 0.03333333333333333, 0: 0.03333333333333333, 1: 0.03333333333333333,
        2: 0.03333333333333333, 3: 0.03333333333333333, 5: 0.06666666666666667, 6: 0.06666666666666667,
        7: 0.06666666666666667, 8: 0.06666666666666667, 9: 0.06666666666666667, 10: 0.06666666666666667,
        11: 0.06666666666666667, 12: 0.06666666666666667, 13: 0.06666666666666667, 14: 0.06666666666666667},
    5: {-5: 0.03333333333333333, -4: 0.03333333333333333, -3: 0.03333333333333333, -2: 0.03333333333333333,
        -1: 0.03333333333333333, 0: 0.03333333333333333, 1: 0.03333333333333333, 2: 0.03333333333333333,
        3: 0.03333333333333333, 4: 0.03333333333333333, 6: 0.06666666666666667, 7: 0.06666666666666667,
        8: 0.06666666666666667, 9: 0.06666666666666667, 10: 0.06666666666666667, 11: 0.06666666666666667,
        12: 0.06666666666666667, 13: 0.06666666666666667, 14: 0.06666666666666667, 15: 0.06666666666666667},
    6: {-4: 0.03333333333333333, -3: 0.03333333333333333, -2: 0.03333333333333333, -1: 0.03333333333333333,
        0: 0.03333333333333333, 1: 0.03333333333333333, 2: 0.03333333333333333, 3: 0.03333333333333333,
        4: 0.03333333333333333, 5: 0.03333333333333333, 7: 0.06666666666666667, 8: 0.06666666666666667,
        9: 0.06666666666666667, 10: 0.06666666666666667, 11: 0.06666666666666667, 12: 0.06666666666666667,
        13: 0.06666666666666667, 14: 0.06666666666666667, 15: 0.06666666666666667, 16: 0.06666666666666667},
    7: {-3: 0.03333333333333333, -2: 0.03333333333333333, -1: 0.03333333333333333, 0: 0.03333333333333333,
        1: 0.03333333333333333, 2: 0.03333333333333333, 3: 0.03333333333333333, 4: 0.03333333333333333,
        5: 0.03333333333333333, 6: 0.03333333333333333, 8: 0.06666666666666667, 9: 0.06666666666666667,
        10: 0.06666666666666667, 11: 0.06666666666666667, 12: 0.06666666666666667, 13: 0.06666666666666667,
        14: 0.06666666666666667, 15: 0.06666666666666667, 16: 0.06666666666666667, 17: 0.06666666666666667},
    8: {-2: 0.03333333333333333, -1: 0.03333333333333333, 0: 0.03333333333333333, 1: 0.03333333333333333,
        2: 0.03333333333333333, 3: 0.03333333333333333, 4: 0.03333333333333333, 5: 0.03333333333333333,
        6: 0.03333333333333333, 7: 0.03333333333333333, 9: 0.06666666666666667, 10: 0.06666666666666667,
        11: 0.06666666666666667, 12: 0.06666666666666667, 13: 0.06666666666666667, 14: 0.06666666666666667,
        15: 0.06666666666666667, 16: 0.06666666666666667, 17: 0.06666666666666667, 18: 0.06666666666666667},
    9: {-1: 0.03333333333333333, 0: 0.03333333333333333, 1: 0.03333333333333333, 2: 0.03333333333333333,
        3: 0.03333333333333333, 4: 0.03333333333333333, 5: 0.03333333333333333, 6: 0.03333333333333333,
        7: 0.03333333333333333, 8: 0.03333333333333333, 10: 0.06666666666666667, 11: 0.06666666666666667,
        12: 0.06666666666666667, 13: 0.06666666666666667, 14: 0.06666666666666667, 15: 0.06666666666666667,
        16: 0.06666666666666667, 17: 0.06666666666666667, 18: 0.06666666666666667, 19: 0.06666666666666667},
    10: {0: 0.03333333333333333, 1: 0.03333333333333333, 2: 0.03333333333333333, 3: 0.03333333333333333,
         4: 0.03333333333333333, 5: 0.03333333333333333, 6: 0.03333333333333333, 7: 0.03333333333333333,
         8: 0.03333333333333333, 9: 0.03333333333333333, 11: 0.06666666666666667, 12: 0.06666666666666667,
         13: 0.06666666666666667, 14: 0.06666666666666667, 15: 0.06666666666666667, 16: 0.06666666666666667,
         17: 0.06666666666666667, 18: 0.06666666666666667, 19: 0.06666666666666667, 20: 0.06666666666666667},
    11: {1: 0.03333333333333333, 2: 0.03333333333333333, 3: 0.03333333333333333, 4: 0.03333333333333333,
         5: 0.03333333333333333, 6: 0.03333333333333333, 7: 0.03333333333333333, 8: 0.03333333333333333,
         9: 0.03333333333333333, 10: 0.03333333333333333, 12: 0.06666666666666667, 13: 0.06666666666666667,
         14: 0.06666666666666667, 15: 0.06666666666666667, 16: 0.06666666666666667, 17: 0.06666666666666667,
         18: 0.06666666666666667, 19: 0.06666666666666667, 20: 0.06666666666666667, 21: 0.06666666666666667},
    12: {2: 0.03333333333333333, 3: 0.03333333333333333, 4: 0.03333333333333333, 5: 0.03333333333333333,
         6: 0.03333333333333333, 7: 0.03333333333333333, 8: 0.03333333333333333, 9: 0.03333333333333333,
         10: 0.03333333333333333, 11: 0.03333333333333333, 13: 0.06666666666666667, 14: 0.06666666666666667,
         15: 0.06666666666666667, 16: 0.06666666666666667, 17: 0.06666666666666667, 18: 0.06666666666666667,
         19: 0.06666666666666667, 20: 0.06666666666666667, 21: 0.06666666666666667, 22: 0.06666666666666667},
    13: {3: 0.03333333333333333, 4: 0.03333333333333333, 5: 0.03333333333333333, 6: 0.03333333333333333,
         7: 0.03333333333333333, 8: 0.03333333333333333, 9: 0.03333333333333333, 10: 0.03333333333333333,
         11: 0.03333333333333333, 12: 0.03333333333333333, 14: 0.06666666666666667, 15: 0.06666666666666667,
         16: 0.06666666666666667, 17: 0.06666666666666667, 18: 0.06666666666666667, 19: 0.06666666666666667,
         20: 0.06666666666666667, 21: 0.06666666666666667, 22: 0.06666666666666667, 23: 0.06666666666666667},
    14: {4: 0.03333333333333333, 5: 0.03333333333333333, 6: 0.03333333333333333, 7: 0.03333333333333333,
         8: 0.03333333333333333, 9: 0.03333333333333333, 10: 0.03333333333333333, 11: 0.03333333333333333,
         12: 0.03333333333333333, 13: 0.03333333333333333, 15: 0.06666666666666667, 16: 0.06666666666666667,
         17: 0.06666666666666667, 18: 0.06666666666666667, 19: 0.06666666666666667, 20: 0.06666666666666667,
         21: 0.06666666666666667, 22: 0.06666666666666667, 23: 0.06666666666666667, 24: 0.06666666666666667},
    15: {5: 0.03333333333333333, 6: 0.03333333333333333, 7: 0.03333333333333333, 8: 0.03333333333333333,
         9: 0.03333333333333333, 10: 0.03333333333333333, 11: 0.03333333333333333, 12: 0.03333333333333333,
         13: 0.03333333333333333, 14: 0.03333333333333333, 16: 0.06666666666666667, 17: 0.06666666666666667,
         18: 0.06666666666666667, 19: 0.06666666666666667, 20: 0.06666666666666667, 21: 0.06666666666666667,
         22: 0.06666666666666667, 23: 0.06666666666666667, 24: 0.06666666666666667, 25: 0.06666666666666667},
    16: {6: 0.03333333333333333, 7: 0.03333333333333333, 8: 0.03333333333333333, 9: 0.03333333333333333,
         10: 0.03333333333333333, 11: 0.03333333333333333, 12: 0.03333333333333333, 13: 0.03333333333333333,
         14: 0.03333333333333333, 15: 0.03333333333333333, 17: 0.06666666666666667, 18: 0.06666666666666667,
         19: 0.06666666666666667, 20: 0.06666666666666667, 21: 0.06666666666666667, 22: 0.06666666666666667,
         23: 0.06666666666666667, 24: 0.06666666666666667, 25: 0.06666666666666667, 26: 0.06666666666666667},
    17: {7: 0.03333333333333333, 8: 0.03333333333333333, 9: 0.03333333333333333, 10: 0.03333333333333333,
         11: 0.03333333333333333, 12: 0.03333333333333333, 13: 0.03333333333333333, 14: 0.03333333333333333,
         15: 0.03333333333333333, 16: 0.03333333333333333, 18: 0.06666666666666667, 19: 0.06666666666666667,
         20: 0.06666666666666667, 21: 0.06666666666666667, 22: 0.06666666666666667, 23: 0.06666666666666667,
         24: 0.06666666666666667, 25: 0.06666666666666667, 26: 0.06666666666666667, 27: 0.06666666666666667},
    18: {8: 0.03333333333333333, 9: 0.03333333333333333, 10: 0.03333333333333333, 11: 0.03333333333333333,
         12: 0.03333333333333333, 13: 0.03333333333333333, 14: 0.03333333333333333, 15: 0.03333333333333333,
         16: 0.03333333333333333, 17: 0.03333333333333333, 19: 0.06666666666666667, 20: 0.06666666666666667,
         21: 0.06666666666666667, 22: 0.06666666666666667, 23: 0.06666666666666667, 24: 0.06666666666666667,
         25: 0.06666666666666667, 26: 0.06666666666666667, 27: 0.06666666666666667, 28: 0.06666666666666667},
    19: {9: 0.03333333333333333, 10: 0.03333333333333333, 11: 0.03333333333333333, 12: 0.03333333333333333,
         13: 0.03333333333333333, 14: 0.03333333333333333, 15: 0.03333333333333333, 16: 0.03333333333333333,
         17: 0.03333333333333333, 18: 0.03333333333333333, 20: 0.06666666666666667, 21: 0.06666666666666667,
         22: 0.06666666666666667, 23: 0.06666666666666667, 24: 0.06666666666666667, 25: 0.06666666666666667,
         26: 0.06666666666666667, 27: 0.06666666666666667, 28: 0.06666666666666667, 29: 0.06666666666666667},
    20: {10: 0.03333333333333333, 11: 0.03333333333333333, 12: 0.03333333333333333, 13: 0.03333333333333333,
         14: 0.03333333333333333, 15: 0.03333333333333333, 16: 0.03333333333333333, 17: 0.03333333333333333,
         18: 0.03333333333333333, 19: 0.03333333333333333, 21: 0.06666666666666667, 22: 0.06666666666666667,
         23: 0.06666666666666667, 24: 0.06666666666666667, 25: 0.06666666666666667, 26: 0.06666666666666667,
         27: 0.06666666666666667, 28: 0.06666666666666667, 29: 0.06666666666666667, 30: 0.06666666666666667},
    21: {11: 0.03333333333333333, 12: 0.03333333333333333, 13: 0.03333333333333333, 14: 0.03333333333333333,
         15: 0.03333333333333333, 16: 0.03333333333333333, 17: 0.03333333333333333, 18: 0.03333333333333333,
         19: 0.03333333333333333, 20: 0.03333333333333333, 22: 0.06666666666666667, 23: 0.06666666666666667,
         24: 0.06666666666666667, 25: 0.06666666666666667, 26: 0.06666666666666667, 27: 0.06666666666666667,
         28: 0.06666666666666667, 29: 0.06666666666666667, 30: 0.06666666666666667, 31: 0.06666666666666667}}

state_trans_stick_reward_2_prob = {
    1: {1: {-1: 0.387, 0: 0.0, 1: 0.613}, 2: {-1: 0.345, 0: 0.0, 1: 0.655}, 3: {-1: 0.397, 0: 0.0, 1: 0.603},
        4: {-1: 0.408, 0: 0.0, 1: 0.592}, 5: {-1: 0.379, 0: 0.0, 1: 0.621}, 6: {-1: 0.375, 0: 0.0, 1: 0.625},
        7: {-1: 0.343, 0: 0.0, 1: 0.657}, 8: {-1: 0.382, 0: 0.0, 1: 0.618}, 9: {-1: 0.394, 0: 0.0, 1: 0.606},
        10: {-1: 0.406, 0: 0.0, 1: 0.594}, 11: {-1: 0.396, 0: 0.0, 1: 0.604}, 12: {-1: 0.373, 0: 0.0, 1: 0.627},
        13: {-1: 0.394, 0: 0.0, 1: 0.606}, 14: {-1: 0.397, 0: 0.0, 1: 0.603}, 15: {-1: 0.38, 0: 0.0, 1: 0.62},
        16: {-1: 0.294, 0: 0.094, 1: 0.612}, 17: {-1: 0.208, 0: 0.081, 1: 0.711}, 18: {-1: 0.143, 0: 0.06, 1: 0.797},
        19: {-1: 0.092, 0: 0.059, 1: 0.849}, 20: {-1: 0.049, 0: 0.046, 1: 0.905}, 21: {-1: 0.0, 0: 0.035, 1: 0.965}},
    2: {1: {-1: 0.419, 0: 0.0, 1: 0.581}, 2: {-1: 0.438, 0: 0.0, 1: 0.562}, 3: {-1: 0.422, 0: 0.0, 1: 0.578},
        4: {-1: 0.392, 0: 0.0, 1: 0.608}, 5: {-1: 0.394, 0: 0.0, 1: 0.606}, 6: {-1: 0.386, 0: 0.0, 1: 0.614},
        7: {-1: 0.406, 0: 0.0, 1: 0.594}, 8: {-1: 0.398, 0: 0.0, 1: 0.602}, 9: {-1: 0.435, 0: 0.0, 1: 0.565},
        10: {-1: 0.444, 0: 0.0, 1: 0.556}, 11: {-1: 0.429, 0: 0.0, 1: 0.571}, 12: {-1: 0.435, 0: 0.0, 1: 0.565},
        13: {-1: 0.39, 0: 0.0, 1: 0.61}, 14: {-1: 0.45, 0: 0.0, 1: 0.55}, 15: {-1: 0.4, 0: 0.0, 1: 0.6},
        16: {-1: 0.346, 0: 0.078, 1: 0.576}, 17: {-1: 0.232, 0: 0.099, 1: 0.669}, 18: {-1: 0.164, 0: 0.071, 1: 0.765},
        19: {-1: 0.104, 0: 0.061, 1: 0.835}, 20: {-1: 0.041, 0: 0.052, 1: 0.907}, 21: {-1: 0.0, 0: 0.039, 1: 0.961}},
    3: {1: {-1: 0.447, 0: 0.0, 1: 0.553}, 2: {-1: 0.45, 0: 0.0, 1: 0.55}, 3: {-1: 0.493, 0: 0.0, 1: 0.507},
        4: {-1: 0.442, 0: 0.0, 1: 0.558}, 5: {-1: 0.478, 0: 0.0, 1: 0.522}, 6: {-1: 0.457, 0: 0.0, 1: 0.543},
        7: {-1: 0.456, 0: 0.0, 1: 0.544}, 8: {-1: 0.457, 0: 0.0, 1: 0.543}, 9: {-1: 0.488, 0: 0.0, 1: 0.512},
        10: {-1: 0.437, 0: 0.0, 1: 0.563}, 11: {-1: 0.437, 0: 0.0, 1: 0.563}, 12: {-1: 0.433, 0: 0.0, 1: 0.567},
        13: {-1: 0.473, 0: 0.0, 1: 0.527}, 14: {-1: 0.418, 0: 0.0, 1: 0.582}, 15: {-1: 0.405, 0: 0.0, 1: 0.595},
        16: {-1: 0.35, 0: 0.101, 1: 0.549}, 17: {-1: 0.231, 0: 0.11, 1: 0.659}, 18: {-1: 0.158, 0: 0.07, 1: 0.772},
        19: {-1: 0.128, 0: 0.068, 1: 0.804}, 20: {-1: 0.044, 0: 0.057, 1: 0.899}, 21: {-1: 0.0, 0: 0.048, 1: 0.952}},
    4: {1: {-1: 0.476, 0: 0.0, 1: 0.524}, 2: {-1: 0.479, 0: 0.0, 1: 0.521}, 3: {-1: 0.459, 0: 0.0, 1: 0.541},
        4: {-1: 0.464, 0: 0.0, 1: 0.536}, 5: {-1: 0.468, 0: 0.0, 1: 0.532}, 6: {-1: 0.483, 0: 0.0, 1: 0.517},
        7: {-1: 0.504, 0: 0.0, 1: 0.496}, 8: {-1: 0.456, 0: 0.0, 1: 0.544}, 9: {-1: 0.503, 0: 0.0, 1: 0.497},
        10: {-1: 0.459, 0: 0.0, 1: 0.541}, 11: {-1: 0.455, 0: 0.0, 1: 0.545}, 12: {-1: 0.467, 0: 0.0, 1: 0.533},
        13: {-1: 0.477, 0: 0.0, 1: 0.523}, 14: {-1: 0.486, 0: 0.0, 1: 0.514}, 15: {-1: 0.472, 0: 0.0, 1: 0.528},
        16: {-1: 0.391, 0: 0.095, 1: 0.514}, 17: {-1: 0.292, 0: 0.09, 1: 0.618}, 18: {-1: 0.187, 0: 0.08, 1: 0.733},
        19: {-1: 0.107, 0: 0.077, 1: 0.816}, 20: {-1: 0.05, 0: 0.071, 1: 0.879}, 21: {-1: 0.0, 0: 0.057, 1: 0.943}},
    5: {1: {-1: 0.534, 0: 0.0, 1: 0.466}, 2: {-1: 0.461, 0: 0.0, 1: 0.539}, 3: {-1: 0.492, 0: 0.0, 1: 0.508},
        4: {-1: 0.467, 0: 0.0, 1: 0.533}, 5: {-1: 0.495, 0: 0.0, 1: 0.505}, 6: {-1: 0.51, 0: 0.0, 1: 0.49},
        7: {-1: 0.52, 0: 0.0, 1: 0.48}, 8: {-1: 0.489, 0: 0.0, 1: 0.511}, 9: {-1: 0.504, 0: 0.0, 1: 0.496},
        10: {-1: 0.498, 0: 0.0, 1: 0.502}, 11: {-1: 0.505, 0: 0.0, 1: 0.495}, 12: {-1: 0.527, 0: 0.0, 1: 0.473},
        13: {-1: 0.483, 0: 0.0, 1: 0.517}, 14: {-1: 0.478, 0: 0.0, 1: 0.522}, 15: {-1: 0.51, 0: 0.0, 1: 0.49},
        16: {-1: 0.403, 0: 0.104, 1: 0.493}, 17: {-1: 0.296, 0: 0.092, 1: 0.612}, 18: {-1: 0.196, 0: 0.095, 1: 0.709},
        19: {-1: 0.131, 0: 0.077, 1: 0.792}, 20: {-1: 0.056, 0: 0.059, 1: 0.885}, 21: {-1: 0.0, 0: 0.06, 1: 0.94}},
    6: {1: {-1: 0.563, 0: 0.0, 1: 0.437}, 2: {-1: 0.553, 0: 0.0, 1: 0.447}, 3: {-1: 0.564, 0: 0.0, 1: 0.436},
        4: {-1: 0.531, 0: 0.0, 1: 0.469}, 5: {-1: 0.577, 0: 0.0, 1: 0.423}, 6: {-1: 0.561, 0: 0.0, 1: 0.439},
        7: {-1: 0.576, 0: 0.0, 1: 0.424}, 8: {-1: 0.477, 0: 0.0, 1: 0.523}, 9: {-1: 0.528, 0: 0.0, 1: 0.472},
        10: {-1: 0.562, 0: 0.0, 1: 0.438}, 11: {-1: 0.542, 0: 0.0, 1: 0.458}, 12: {-1: 0.568, 0: 0.0, 1: 0.432},
        13: {-1: 0.537, 0: 0.0, 1: 0.463}, 14: {-1: 0.55, 0: 0.0, 1: 0.45}, 15: {-1: 0.524, 0: 0.0, 1: 0.476},
        16: {-1: 0.371, 0: 0.164, 1: 0.465}, 17: {-1: 0.256, 0: 0.104, 1: 0.64}, 18: {-1: 0.176, 0: 0.076, 1: 0.748},
        19: {-1: 0.12, 0: 0.075, 1: 0.805}, 20: {-1: 0.063, 0: 0.063, 1: 0.874}, 21: {-1: 0.0, 0: 0.052, 1: 0.948}},
    7: {1: {-1: 0.623, 0: 0.0, 1: 0.377}, 2: {-1: 0.621, 0: 0.0, 1: 0.379}, 3: {-1: 0.622, 0: 0.0, 1: 0.378},
        4: {-1: 0.595, 0: 0.0, 1: 0.405}, 5: {-1: 0.608, 0: 0.0, 1: 0.392}, 6: {-1: 0.59, 0: 0.0, 1: 0.41},
        7: {-1: 0.581, 0: 0.0, 1: 0.419}, 8: {-1: 0.604, 0: 0.0, 1: 0.396}, 9: {-1: 0.57, 0: 0.0, 1: 0.43},
        10: {-1: 0.587, 0: 0.0, 1: 0.413}, 11: {-1: 0.591, 0: 0.0, 1: 0.409}, 12: {-1: 0.599, 0: 0.0, 1: 0.401},
        13: {-1: 0.597, 0: 0.0, 1: 0.403}, 14: {-1: 0.583, 0: 0.0, 1: 0.417}, 15: {-1: 0.596, 0: 0.0, 1: 0.404},
        16: {-1: 0.427, 0: 0.168, 1: 0.405}, 17: {-1: 0.275, 0: 0.183, 1: 0.542}, 18: {-1: 0.185, 0: 0.095, 1: 0.72},
        19: {-1: 0.113, 0: 0.067, 1: 0.82}, 20: {-1: 0.059, 0: 0.057, 1: 0.884}, 21: {-1: 0.0, 0: 0.069, 1: 0.931}},
    8: {1: {-1: 0.64, 0: 0.0, 1: 0.36}, 2: {-1: 0.626, 0: 0.0, 1: 0.374}, 3: {-1: 0.648, 0: 0.0, 1: 0.352},
        4: {-1: 0.636, 0: 0.0, 1: 0.364}, 5: {-1: 0.618, 0: 0.0, 1: 0.382}, 6: {-1: 0.64, 0: 0.0, 1: 0.36},
        7: {-1: 0.64, 0: 0.0, 1: 0.36}, 8: {-1: 0.599, 0: 0.0, 1: 0.401}, 9: {-1: 0.634, 0: 0.0, 1: 0.366},
        10: {-1: 0.649, 0: 0.0, 1: 0.351}, 11: {-1: 0.636, 0: 0.0, 1: 0.364}, 12: {-1: 0.603, 0: 0.0, 1: 0.397},
        13: {-1: 0.628, 0: 0.0, 1: 0.372}, 14: {-1: 0.62, 0: 0.0, 1: 0.38}, 15: {-1: 0.632, 0: 0.0, 1: 0.368},
        16: {-1: 0.46, 0: 0.175, 1: 0.365}, 17: {-1: 0.319, 0: 0.155, 1: 0.526}, 18: {-1: 0.179, 0: 0.152, 1: 0.669},
        19: {-1: 0.107, 0: 0.081, 1: 0.812}, 20: {-1: 0.044, 0: 0.066, 1: 0.89}, 21: {-1: 0.0, 0: 0.059, 1: 0.941}},
    9: {1: {-1: 0.676, 0: 0.0, 1: 0.324}, 2: {-1: 0.706, 0: 0.0, 1: 0.294}, 3: {-1: 0.66, 0: 0.0, 1: 0.34},
        4: {-1: 0.648, 0: 0.0, 1: 0.352}, 5: {-1: 0.65, 0: 0.0, 1: 0.35}, 6: {-1: 0.698, 0: 0.0, 1: 0.302},
        7: {-1: 0.667, 0: 0.0, 1: 0.333}, 8: {-1: 0.666, 0: 0.0, 1: 0.334}, 9: {-1: 0.69, 0: 0.0, 1: 0.31},
        10: {-1: 0.655, 0: 0.0, 1: 0.345}, 11: {-1: 0.683, 0: 0.0, 1: 0.317}, 12: {-1: 0.684, 0: 0.0, 1: 0.316},
        13: {-1: 0.689, 0: 0.0, 1: 0.311}, 14: {-1: 0.665, 0: 0.0, 1: 0.335}, 15: {-1: 0.648, 0: 0.0, 1: 0.352},
        16: {-1: 0.525, 0: 0.155, 1: 0.32}, 17: {-1: 0.397, 0: 0.14, 1: 0.463}, 18: {-1: 0.261, 0: 0.16, 1: 0.579},
        19: {-1: 0.101, 0: 0.133, 1: 0.766}, 20: {-1: 0.049, 0: 0.065, 1: 0.886}, 21: {-1: 0.0, 0: 0.051, 1: 0.949}},
    10: {1: {-1: 0.725, 0: 0.0, 1: 0.275}, 2: {-1: 0.738, 0: 0.0, 1: 0.262}, 3: {-1: 0.719, 0: 0.0, 1: 0.281},
         4: {-1: 0.715, 0: 0.0, 1: 0.285}, 5: {-1: 0.732, 0: 0.0, 1: 0.268}, 6: {-1: 0.737, 0: 0.0, 1: 0.263},
         7: {-1: 0.718, 0: 0.0, 1: 0.282}, 8: {-1: 0.739, 0: 0.0, 1: 0.261}, 9: {-1: 0.713, 0: 0.0, 1: 0.287},
         10: {-1: 0.73, 0: 0.0, 1: 0.27}, 11: {-1: 0.703, 0: 0.0, 1: 0.297}, 12: {-1: 0.719, 0: 0.0, 1: 0.281},
         13: {-1: 0.725, 0: 0.0, 1: 0.275}, 14: {-1: 0.734, 0: 0.0, 1: 0.266}, 15: {-1: 0.711, 0: 0.0, 1: 0.289},
         16: {-1: 0.573, 0: 0.156, 1: 0.271}, 17: {-1: 0.427, 0: 0.151, 1: 0.422}, 18: {-1: 0.297, 0: 0.157, 1: 0.546},
         19: {-1: 0.166, 0: 0.139, 1: 0.695}, 20: {-1: 0.04, 0: 0.13, 1: 0.83}, 21: {-1: 0.0, 0: 0.046, 1: 0.954}}}
