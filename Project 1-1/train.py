import os
import types
import numpy as np
from tqdm import tqdm

import constants
import environment
import policy
import update


def check_output_path(output_path: str) -> bool:
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path):
        raise RuntimeError("[Error] Assigned Output Path too Deep")

    return True


class TrainQTable:
    def __init__(self,
                 train_epoch: int = 10000,
                 policy_func: types.MethodType = None,
                 update_func: types.MethodType = None,
                 output_path: str = "./_trained/Q_Learning/",
                 progress_bar: bool = True):
        self.train_epoch = train_epoch
        self.env_obj = environment.Easy21Env()
        self.policy_func = policy_func if policy_func else policy.ActionPolicies().greedy_epsilon
        self.update_func = update_func if update_func \
            else update.UpdateQTable().q_function  # learning_rate=0.1, discount_factor=0.5
        self.progress_bar = progress_bar

        _ = check_output_path(output_path=output_path)
        self.output_path = output_path

    def train(self, epsilon: float = 0.5,
              init_filename: str = None, filename: str = "TestOutput") -> None:
        # initiate
        if init_filename is not None and os.path.exists(init_filename):
            q_table = np.load(init_filename)
        else:
            q_table = np.zeros(constants.STATE_ACTION_SPACE_SHAPE)  # Q-Table

        iter_obj = tqdm(range(self.train_epoch)) if self.progress_bar else range(self.train_epoch)
        for ep_idx in iter_obj:
            self.env_obj.reset()
            state_current = self.env_obj.observe()
            while not constants.judge_state_is_terminate(state_current):
                action = self.policy_func(
                    q_table=q_table, state=state_current, epsilon=epsilon)
                state_next, reward, _ = self.env_obj.step(action=action)
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
        print("Trained Model Saved:", self.output_path + filename)


class TrainPolicyIteration:
    def __init__(self,
                 update_obj: update.PolicyIterationUpdates = None,
                 output_path: str = "./_trained/Policy_Iteration/"):
        self.update_obj = update_obj if update_obj is not None else \
            update.PolicyIterationUpdates(
                state_trans_hit_prob=constants.state_trans_hit_prob,
                state_trans_stick_reward_2_prob=constants.state_trans_stick_reward_2_prob,
                discount_factor=0.5)

        _ = check_output_path(output_path=output_path)
        self.output_path = output_path

    def train(self, init_table_value_filename: str = None,
              init_table_action_filename: str = None,
              filename: str = "TestOutput") -> None:
        # initiate
        if init_table_value_filename is not None and os.path.exists(init_table_value_filename):
            table_value = np.load(init_table_value_filename)
        else:
            table_value = np.zeros(constants.STATE_SPACE_SHAPE)  # Value Table
        if init_table_action_filename is not None and os.path.exists(init_table_action_filename):
            table_action = np.load(init_table_action_filename)
        else:
            table_action = np.zeros(constants.STATE_SPACE_SHAPE)  # Value Table

        policy_is_stable = False
        while not policy_is_stable:
            table_value, table_value_is_converged, table_value_update_iter_cnt = \
                self.update_obj.policy_evaluation(
                    table_value=table_value, table_action=table_action,
                    delta_thres=1e-3, max_iter_cnt=1000)

            table_action, policy_is_stable = self.update_obj.policy_improvement(
                table_value=table_value, table_action=table_action)

        # print(q_table)
        np.save(file=self.output_path + filename, arr=table_action)
        print("Trained Model Saved:", self.output_path + filename)


if "__main__" == __name__:
    # # Q-Learning
    # learning_rate_values = [0.7, ]
    # discount_factor_values = [1, ]
    # epsilon_values = [0.6, ]
    # for learning_rate in learning_rate_values:
    #     for discount_factor in discount_factor_values:
    #         test_update_obj = update.UpdateQTable(
    #             learning_rate=learning_rate, discount_factor=discount_factor)
    #         test_update_func = test_update_obj.q_function
    #
    #         for _epsilon in epsilon_values:
    #             test_policy_func = policy.ActionPolicies().greedy_epsilon
    #             train_obj = TrainQTable(**{
    #                 "policy_func": test_policy_func,
    #                 "update_func": test_update_func,
    #                 "progress_bar": False})
    #             train_obj.train()

    # Policy Iteration
    discount_factor_values = [1, ]
    for discount_factor in discount_factor_values:
        test_update_obj = update.PolicyIterationUpdates(
            state_trans_hit_prob=constants.state_trans_hit_prob,
            state_trans_stick_reward_2_prob=constants.state_trans_stick_reward_2_prob,
            discount_factor=discount_factor)
        train_obj = TrainPolicyIteration(update_obj=test_update_obj)
        train_obj.train()
