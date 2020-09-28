import os
import types
import numpy as np

import constants
import environment
import policy
import update


class EvaluateQTable:
    def __init__(self,
                 policy_func: types.MethodType = None,
                 update_func: types.MethodType = None,
                 trained_path: str = "./_trained/"):
        self.env_obj = environment.Easy21Env()
        self.policy_func = policy_func if policy_func else policy.ActionPolicies().greedy_epsilon
        self.update_func = update_func if update_func \
            else update.UpdateQTable().q_function  # learning_rate=0.1, discount_factor=0.5

        self.trained_path = trained_path
        if not os.path.exists(trained_path):
            raise RuntimeError("[Error] Trained Path NOT Found")

    def evaluate(self, epsilon: float = 0.5, filename: str = "TestOutput.npy") -> None:
        # initiate
        q_table = np.load(os.path.join(self.trained_path, filename))  # Q-Table

        self.env_obj.reset()
        state_current = self.env_obj.observe()
        while not constants.judge_state_is_terminate(state_current):
            action = self.policy_func(
                q_table=q_table, state=state_current, epsilon=epsilon)
            state_next, reward, card = self.env_obj.step(action=action)

            print("{:15}\tDealer={}, Player={}".format("[CURRENT STATE]", state_current[0], state_current[1]))
            print("{:15}\t{}".format("[ACTION]", "STICK" if action else "HIT"))
            if isinstance(card["color"], str):  # single card
                print("{:15}\t{}{}".format("[CARD]", {"RED": "-", "BLACK": "+"}[card["color"]], card["value"]))
            else:  # single/multiple card(s)
                print("{:15}\t{}".format("[CARD]",
                                         ", ".join(["(%s%d)" % ({"RED": "-", "BLACK": "+"}[_cd[0]], _cd[1])
                                                    for _cd in zip(card["color"], card["value"])])))
            print("{:15}\tDealer={}, Player={}".format("[NEXT STATE]", state_next[0], state_next[1]))
            print("{:15}\t{}".format("[REWARD]", reward))
            print()

            # update state
            state_current = state_next


if "__main__" == __name__:
    learning_rate_values = [0.7, ]
    discount_factor_values = [1, ]
    epsilon_values = [0.6, ]
    for learning_rate in learning_rate_values:
        for discount_factor in discount_factor_values:
            test_update_obj = update.UpdateQTable(
                learning_rate=learning_rate, discount_factor=discount_factor)
            test_update_func = test_update_obj.q_function

            for _epsilon in epsilon_values:
                test_policy_func = policy.ActionPolicies().greedy_epsilon
                eval_obj = EvaluateQTable(**{"policy_func": test_policy_func, "update_func": test_update_func})
                eval_obj.evaluate()
