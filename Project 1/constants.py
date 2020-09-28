DECK = range(1, 10 + 1)
ACTIONS = (HIT, STICK) = (0, 1)

DEALER_RANGE = range(1, 10 + 1)  # value of the dealer's first card
PLAYER_RANGE = range(1, 21 + 1)
STATE_SPACE_SHAPE = (len(DEALER_RANGE), len(PLAYER_RANGE), len(ACTIONS))

TERMINAL_STATE = "TERMINAL"
DEALER_STICK_THRES = 16
COLOR_PROBS = {"red": 1 / 3, "black": 2 / 3}
COLOR_COEFFS = {"red": -1, "black": 1}