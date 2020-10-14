import os
import types
import typing
import numpy as np

import constants
import environment
import policy
import update


class EvaluateEnv:
    def __init__(self, show_details: bool = True):
        self.env_obj = environment.Easy21Env()
        self.show_details = show_details

    def reset(self):
        self.env_obj.reset()

    def evaluate(self, action_func: types.MethodType) -> int:
        self.reset()
        state_current = self.env_obj.observe()

        reward = -99
        if self.show_details:
            print("=============================================", end="")
        while not constants.judge_state_is_terminate(state_current):
            action = action_func(state=state_current)
            state_next, reward, card = self.env_obj.step(action=action)

            if self.show_details:
                print()
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

            # update state
            state_current = state_next
        if self.show_details:
            print("=============================================", end="\n\n")

        return reward


class QLearningEnv:
    def __init__(self,
                 policy_func: types.MethodType = None,
                 trained_path: str = "./_trained/Q_Learning/"):
        self.policy_func = policy_func if policy_func else policy.ActionPolicies().greedy_maximum

        self.trained_path = trained_path
        if not os.path.exists(trained_path):
            raise RuntimeError("[Error] Trained Path NOT Found")

        # Model Related Initialization
        self.q_table = None
        self.epsilon = None

    def load_model(self, epsilon: float = 0.5, filename: str = "TestOutput.npy") -> None:
        self.q_table = np.load(os.path.join(self.trained_path, filename))  # Q-Table
        self.epsilon = epsilon

    def action_func(self, state: typing.Tuple[int, int]) -> int:
        action = self.policy_func(
            q_table=self.q_table, state=state, epsilon=self.epsilon)

        return action


class PolicyIterationEnv:
    def __init__(self, trained_path: str = "./_trained/Policy_Iteration/"):
        self.trained_path = trained_path
        if not os.path.exists(trained_path):
            raise RuntimeError("[Error] Trained Path NOT Found")

        # Model Related Initialization
        self.table_policy = None

    def load_model(self, filename: str = "TestOutput.npy") -> None:
        self.table_policy = np.load(os.path.join(self.trained_path, filename))  # Policy Table

    def action_func(self, state: typing.Tuple[int, int]) -> int:
        action = self.table_policy[state[0] - 1, state[1] - 1]
        return action


if "__main__" == __name__:
    epsilon_values = [0.6, ]
    evaluate_rounds = 10000
    test_eval_env_obj = EvaluateEnv(show_details=True)

    # # Q-Learning
    # test_policy_func = policy.ActionPolicies().greedy_epsilon
    # test_ql_env_obj = QLearningEnv(**{"policy_func": test_policy_func})
    # for _epsilon in epsilon_values:
    #     test_ql_env_obj.load_model(epsilon=_epsilon, filename="TestOutput.npy")
    #
    #     results = {-1: 0, 0: 0, 1: 0, "err": 0}
    #     for rd in range(evaluate_rounds):
    #         terminate_reward = test_eval_env_obj.evaluate(
    #             **{"action_func": test_ql_env_obj.action_func})
    #         try:
    #             results[terminate_reward] += 1
    #         except KeyError:
    #             results["error"] += 1
    #
    #     print("\n\n\n")
    #     print("WIN / TIE / LOSE / ERR / ALL\n%d / %d / %d / %d / %d" % (
    #         results[1], results[0], results[-1], results["err"], evaluate_rounds))
    #     print("Win Rate: %.2f %%" % (float(results[1]) / evaluate_rounds * 100.))

    # Policy Iteration
    test_policy_func = policy.ActionPolicies().greedy_epsilon
    test_pi_action_obj = PolicyIterationEnv()
    test_pi_action_obj.load_model(filename="TestOutput.npy")

    results = {-1: 0, 0: 0, 1: 0, "err": 0}
    for rd in range(evaluate_rounds):
        terminate_reward = test_eval_env_obj.evaluate(
            **{"action_func": test_pi_action_obj.action_func})
        try:
            results[terminate_reward] += 1
        except KeyError:
            results["error"] += 1

    print("\n\n\n")
    print("WIN / TIE / LOSE / ERR / ALL\n%d / %d / %d / %d / %d" % (
        results[1], results[0], results[-1], results["err"], evaluate_rounds))
    print("Win Rate: %.2f %%" % (float(results[1]) / evaluate_rounds * 100.))
