import numpy as np
import constants


def draw_card(color=None):
    value = np.random.choice(constants.DECK)
    if color is None:
        colors, probs = zip(*constants.COLOR_PROBS.items())
        color = np.random.choice(colors, p=probs)
    return {"value": value, "color": color}


def bust(x):
    return x < 1 or 21 < x


class Easy21Env:
    """ Easy21 environment
    Easy21 is a simple card game similar to Blackjack The rules of the game are as
    follows:
    - The game is played with an infinite deck of cards (i.e. cards are sampled
      with replacement)
    - Each draw from the deck results in a value between 1 and 10 (uniformly
      distributed) with a colour of red (probability 1/3) or black (probability
      2/3).
    - There are no aces or picture (face) cards in this game
    - At the start of the game both the player and the dealer draw one black
      card (fully observed)
    - Each turn the player may either stick or hit
    - If the player hits then she draws another card from the deck
    - If the player sticks she receives no further cards
    - The values of the player's cards are added (black cards) or subtracted (red
      cards)
    - If the player's sum exceeds 21, or becomes less than 1, then she "goes
      bust" and loses the game (reward -1)
    - If the player sticks then the dealer starts taking turns. The dealer always
      sticks on any sum of 15 or greater, and hits otherwise. If the dealer goes
      bust, then the player wins; otherwise, the outcome - win (reward +1),
      lose (reward -1), or draw (reward 0) - is the player with the largest sum.
    """

    def __init__(self):
        self.dealer, self.player = None, None  # values of the dealer / player
        self.reset()

    def reset(self, dealer=None, player=None):
        if dealer is None:
            dealer = draw_card()["value"]
        self.dealer = dealer
        if player is None:
            player = draw_card()["value"]
        self.player = player

    def observe(self):
        if not (self.dealer in constants.DEALER_RANGE and self.player in constants.PLAYER_RANGE):
            return constants.TERMINAL_STATE
        return np.array((self.dealer, self.player))

    def step(self, action):
        """ Step function
        Inputs:
        - action: hit or stick
        Returns:
        - next_state: a sample of the next state (which may be terminal if the
          game is finished)
        - reward
        """

        # PLAYER CHOOSES HIT (0)
        if action == constants.HIT:
            card = draw_card()
            self.player += constants.COLOR_COEFFS[card["color"]] * card["value"]

            if constants.DEBUG_DEBUG:
                print("Card:", card["color"], card["value"])

            if bust(self.player):
                next_state, reward = (constants.TERMINAL_STATE, constants.TERMINAL_STATE), -1
            else:
                next_state, reward = (self.dealer, self.player), 0

            return np.array(next_state), reward, card

        # PLAYER CHOOSES STICK (1)
        elif action == constants.STICK:
            _cards_history = {"color": [], "value": []}
            while 0 < self.dealer < constants.DEALER_STICK_THRES:
                card = draw_card()
                _cards_history["color"].append(card["color"])
                _cards_history["value"].append(card["value"])
                self.dealer += constants.COLOR_COEFFS[card["color"]] * card["value"]

            next_state = (constants.TERMINAL_STATE, constants.TERMINAL_STATE)
            if bust(self.dealer):
                reward = 1
            else:
                reward = int(self.player > self.dealer) - int(self.player < self.dealer)

            return np.array(next_state), reward, _cards_history

        # ERROR PLAYER ACTION
        else:
            raise ValueError("Action not in action space")
