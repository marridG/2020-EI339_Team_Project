import os
from datetime import datetime
from tqdm import tqdm

import constants
import evaluate
import policy
import update


def eval_param(trained_model_path: str = "./_trained/",
               evaluate_rounds: int = 10000,
               evaluate_result_path: str = "./_trained/eval/"):
    models_filename_lst = os.listdir(trained_model_path)
    models_result = []
    for model_filename in tqdm(models_filename_lst):
        try:
            setting_split_str = os.path.splitext(model_filename)[0].split("_")
            setting = {"epoch": int(setting_split_str[0].split("=")[1]),
                       "learning rate": float(setting_split_str[1].split("=")[1]),
                       "discount factor": float(setting_split_str[2].split("=")[1]),
                       "epsilon": float(setting_split_str[3].split("=")[1]), }
        except Exception as err:
            print("Skipped Train Model \"%s\", cuz %s" % (model_filename, str(err)))
            continue

        test_update_obj = update.UpdateQTable(
            learning_rate=setting["learning rate"],
            discount_factor=setting["discount factor"])
        test_update_func = test_update_obj.q_function

        test_policy_func = policy.ActionPolicies().greedy_epsilon
        eval_obj = evaluate.EvaluateQTable(
            **{"policy_func": test_policy_func,
               "update_func": test_update_func, })

        results = {-1: 0, 0: 0, 1: 0, "err": 0}
        for rd in range(evaluate_rounds):
            terminate_reward = eval_obj.evaluate(
                epsilon=setting["epsilon"], filename=model_filename,
                show_details=False)
            try:
                results[terminate_reward] += 1
            except KeyError:
                results["error"] += 1

        models_result.append({"filename": model_filename, "setting": setting, "result": results})

    max_win_cnt, max_win_rate_setting = 0, None
    with open(os.path.join(evaluate_result_path,
                           "evaluation_report_%s"
                           % datetime.now().strftime("%Y%m%d%H%M%S")), "w") as f:
        for _model in models_result:
            f.write("Filename:%s\t"
                    "Epoch:%d\tLearning Rate:%f\t"
                    "Discount Factor:%f\tEpsilon:%f\t"
                    "WIN:%d\tTIE:%d\tLOSE:%d\tERR:%d\tALL:%d\n" %
                    (_model["filename"],
                     _model["setting"]["epoch"], _model["setting"]["learning rate"],
                     _model["setting"]["discount factor"], _model["setting"]["epsilon"],
                     _model["result"][1], _model["result"][0],
                     _model["result"][-1], _model["result"]["err"],
                     evaluate_rounds))

            if _model["result"][1] > max_win_cnt:
                max_win_cnt = _model["result"][1]
                max_win_rate_setting = _model["setting"]

    print("\nEvaluation Done\n")
    print("[Max Win Count]", max_win_cnt)
    print("[Max Count Setting]", max_win_rate_setting)


if "__main__" == __name__:
    eval_param(trained_model_path="../../_trained/", evaluate_rounds=1000)
