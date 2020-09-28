import os
import types
import numpy as np
from tqdm import tqdm

import constants
import environment
import policy
import update


class TrainQTable:
    def __init__(self,
                 policy_func: types.MethodType = None,
                 update_func: types.MethodType = None,
                 output_path: str = "./_trained/"):
        self.env_obj = environment.Easy21Env()
        self.policy_func = policy_func if policy_func else policy.ActionPolicies().greedy_epsilon
        self.update_func = update_func if update_func \
            else update.UpdateQTable().q_function  # learning_rate=0.1, discount_factor=0.5

        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(output_path):
            raise RuntimeError("[Error] Assigned Output Path too Deep")

    def train(self, epsilon: float = 0.5, filename: str = "TestOutput") -> None:
        # initiate
        q_table = np.zeros(constants.STATE_SPACE_SHAPE)  # Q-Table

        for ep_idx in tqdm(range(constants.TR_EPISODE)):

            self.env_obj.reset()
            state_current = self.env_obj.observe()
            while not constants.judge_state_is_terminate(state_current):
                action = self.policy_func(
                    q_table=q_table, state=state_current, epsilon=epsilon)
                state_next, reward = self.env_obj.step(action=action)
                if constants.DEBUG_INFO:
                    print(state_current, action, state_next, reward)

                # update q-table
                new_val = self.update_func(
                    q_table=q_table, state_crt=state_current, action=action,
                    reward=reward, state_next=state_next)
                if constants.DEBUG_INFO:
                    print(new_val)
                q_table[state_current[0] - 1, state_current[1] - 1, action] = new_val
                # update state
                state_current = state_next

        # print(q_table)
        np.save(file=self.output_path + filename, arr=q_table)


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
                train_obj = TrainQTable(**{"policy_func": test_policy_func, "update_func": test_update_func})
                train_obj.train()
