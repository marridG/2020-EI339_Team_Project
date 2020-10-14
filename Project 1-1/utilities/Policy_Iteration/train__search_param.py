import numpy as np
from tqdm import tqdm

import constants
import train
import update

discount_factor_values = np.linspace(0, 1, 25 + 1)[1:-2]

train_settings = [(df,) for df in discount_factor_values]
epoch = 10000
print("Initiate %d Settings" % len(train_settings))

for setting_idx, (discount_factor,) in tqdm(enumerate(train_settings)):
    test_update_obj = update.PolicyIterationUpdates(
        state_trans_hit_prob=constants.state_trans_hit_prob,
        state_trans_stick_reward_2_prob=constants.state_trans_stick_reward_2_prob,
        discount_factor=discount_factor)
    train_obj = train.TrainPolicyIteration(update_obj=test_update_obj)
    train_obj.train()
