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


class PolicyIterationUpdates:
    def __init__(self,
                 state_trans_hit_prob: typing.Dict[int, typing.Dict[int, float]],
                 state_trans_stick_reward_2_prob: typing.Dict[int, typing.Dict[int, typing.Dict[int, float]]],
                 learning_rate: float = 0.1, discount_factor: float = 0.5):
        self.state_trans_hit_prob = state_trans_hit_prob
        self.state_trans_stick_reward_2_prob = state_trans_stick_reward_2_prob
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def policy_evaluation(self, table_value: np.ndarray,
                          table_action: np.ndarray,
                          delta_thres: float = 1e-3, max_iter_cnt: int = 1000) \
            -> (np.ndarray, bool, int):
        """
        update Value Table
        :param table_value:         Value Table
        :param table_action:        Action Table
        :param delta_thres:         value convergence threshold
        :param max_iter_cnt:        maximum attempts to converge
        :return:                    <np.ndarray> updated Value Table;
                                    <bool> is converged;
                                    <int> iter counts;
        """
        iter_cnt = 0
        while iter_cnt < max_iter_cnt:
            iter_cnt += 1
            delta = 0
            for (dealer, player) in constants.STATE_SPACE:
                state_action = table_action[dealer - 1, player - 1]
                state_value_ori = table_value[dealer - 1, player - 1]

                state_value_new = 1  # todo
                delta = max(delta, abs(state_value_ori - state_value_new))

            if delta < delta_thres:
                return table_value, True, iter_cnt
        return table_value, False, iter_cnt

    def policy_improvement(self):
        pass


if "__main__" == __name__:
    # update_obj = UpdateQTable()  # learning_rate=0.1, discount_factor=0.5
    # test_q_table = np.abs(np.random.randn(*constants.STATE_ACTION_SPACE_SHAPE))
    # test_new_val = update_obj.q_function(
    #     q_table=test_q_table, state_crt=(5, 7), action=0,
    #     reward=0, state_next=(5, 9))
    # print(test_new_val)

    update_obj = PolicyIterationUpdates(state_trans_hit_prob={1: {0: 0.3}},
                                        state_trans_stick_reward_2_prob={1: {1: {-1: 0.3}}},
                                        learning_rate=0.1, discount_factor=0.5)
    test_value_table = np.abs(np.random.randn(*constants.STATE_SPACE_SHAPE))
    test_action_table = np.abs(np.random.randn(*constants.STATE_SPACE_SHAPE))
    test_value_table, test_is_converged, test_iter_cnt = update_obj.policy_evaluation(
        table_value=test_value_table, table_action=test_action_table,
        delta_thres=1e-3, max_iter_cnt=1000)
    print(test_value_table)
    print(test_is_converged)
    print(test_iter_cnt)
