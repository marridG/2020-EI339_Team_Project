import os
import types
import numpy as np
from tqdm import tqdm

import constants
import environment
import train
import policy
import update

learning_rate_values = np.linspace(0, 1, 500 + 1)
discount_factor_values = np.linspace(0, 1, 500 + 1)
epsilon_values = np.linspace(0, 1, 100 + 1)

train_settings = [(lr, df, ep)
                  for lr in learning_rate_values
                  for df in discount_factor_values
                  for ep in epsilon_values]
epoch = 10000
print("Initiate %d Settings" % len(train_settings))

for setting_idx, (learning_rate, discount_factor, epsilon) in tqdm(enumerate(train_settings)):
    test_update_obj = update.UpdateQTable(
        learning_rate=learning_rate, discount_factor=discount_factor)
    test_update_func = test_update_obj.q_function

    test_policy_func = policy.ActionPolicies().greedy_epsilon
    train_obj = train.TrainQTable(**{"train_epoch": epoch,
                                     "policy_func": test_policy_func,
                                     "update_func": test_update_func,
                                     "progress_bar": False})
    train_obj.train(epsilon=epsilon,
                    filename="epoch=%d_lr=%.4f_df=%.4f_ep=%.4f"
                             % (epoch, learning_rate, discount_factor, epsilon))
