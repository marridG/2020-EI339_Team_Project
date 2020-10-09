import os
import numpy as np
from tqdm import tqdm

import constants
import evaluate
import policy
import update


def eval_param(trained_model_path:str="./_trained", evaluate_rounds:int = 10000):
    models_filename_lst = os.listdir(trained_model_path)
    models_result=[]
    for model_filename in tqdm(models_filename_lst):
        try:
            setting_split_str = os.path.splitext(model_filename)[0].split("_")
            setting = {"epoch": int(setting_split_str[0].split("=")[1]),
                       "learning rate": float(setting_split_str[1].split("=")[1]),
                       "discount factor": float(setting_split_str[2].split("=")[1]),
                       "epsilon": float(setting_split_str[0].split("=")[1]),}

            test_update_obj = update.UpdateQTable(
                learning_rate=setting["learning rate"],
                discount_factor=setting["discount factor"])
            test_update_func = test_update_obj.q_function

            test_policy_func = policy.ActionPolicies().greedy_epsilon
            eval_obj = evaluate.EvaluateQTable(
                **{"policy_func": test_policy_func,
                   "update_func": test_update_func,})

            results = {-1: 0, 0: 0, 1: 0, "err": 0}
            for rd in range(evaluate_rounds):
                terminate_reward = eval_obj.evaluate(
                    epsilon=setting["epsilon"],
                    filename=model_filename,
                    show_details=False)
                try:
                    results[terminate_reward] += 1
                except KeyError:
                    results["error"] += 1

            models_result.append([setting,results])

        except Exception as err:
            print("Skipped Train Model %s, cuz %s" % (model_filename, str(err)))
            continue

    max_win_cnt, max_win_rate_setting = 0, None
    for _model in models_result:
        if _model[1][1]>max_win_cnt:
            max_win_cnt=_model[1][1]
            max_win_rate_setting=_model[0]

    print("\n\n")
    print( max_win_cnt)
    print(max_win_rate_setting)
    print(models_result)



if "__main__" == __name__:
    eval_param(trained_model_path="./_trained/",evaluate_rounds=1000)

