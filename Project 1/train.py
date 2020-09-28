import numpy as np

import constants
import environment
import policy
import update

env_obj = environment.Easy21Env()
policy_obj = policy.ActionPolicies()
# update_obj = None
learning_rate_values = [0.1, ]
discount_factor_values = [0.95, ]
epsilon_values = [0.5, ]
for learning_rate in learning_rate_values:
    for discount_factor in discount_factor_values:
        update_obj = update.UpdateQTable(
            learning_rate=learning_rate, discount_factor=discount_factor)

        for epsilon in epsilon_values:
            # initiate
            q_table = np.zeros(constants.STATE_SPACE_SHAPE)  # Q-Table

            for ep_idx in range(constants.TR_EPISODE):
                env_obj.reset()
                state_current = env_obj.observe()
                while not constants.judge_state_is_terminate(state_current):
                    action = policy_obj.greedy_epsilon(
                        q_table=q_table, state=state_current, epsilon=epsilon)
                    state_next, reward = env_obj.step(action=action)
                    if constants.DEBUG_INFO:
                        print(state_current, action, state_next, reward)

                    # update q-table
                    new_val = update_obj.q_function(
                        q_table=q_table, state_crt=state_current, action=action,
                        reward=reward, state_next=state_next)
                    if constants.DEBUG_INFO:
                        print(new_val)
                    q_table[state_current[0] - 1, state_current[1] - 1, action] = new_val
                    # update state
                    state_current = state_next

            print(q_table)
            # np.save(file="lr=%.2f_df=%.2f_ep=%.2f.txt"
            #              % (learning_rate, discount_factor, epsilon),
            #         arr=q_table)
