# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating
import time

# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]

env_id = "Qube-100-v0"  # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
config_path = "config.yml"
config = load_config(config_path)
print_config(config_path)

batchsize_list=[]
total_rewardlist=[]
for i in range(4):
    batchsize_list.append(config["training_config"]["batch_size"])
    model = DynamicModel(config)
    data_fac = DatasetFactory(env, config)
    data_fac.collect_random_dataset()
    loss = model.train(data_fac.random_trainset, data_fac.random_testset)
    mpc = MPC(env, config)
    rewards_list = []
    for itr in range(config["dataset_config"]["n_mpc_itrs"]//2):
        t = time.time()
        print("**********************************************")
        print("The reinforce process [%s], collecting data ..." % itr)
        rewards = data_fac.collect_mpc_dataset(mpc, model)
        trainset, testset = data_fac.make_dataset()
        rewards_list += rewards


        print("Consume %s s in this iteration" % (time.time() - t))
        loss = model.train(trainset, testset)
    config["training_config"]["batch_size"]*=2
    total_rewardlist.append(rewards_list)


print(batchsize_list)
print(total_rewardlist)

plt.close("all")
plt.figure(figsize=(12, 5))
plt.title('Reward Trend with {} iteration' .format(config["dataset_config"]["n_mpc_itrs"]//len(batchsize_list)))
for i in range(len(batchsize_list)):
    plt.plot(total_rewardlist[i],label=("batchsize={}".format(batchsize_list[i])))
plt.legend(loc="upper left")
plt.show()
plt.savefig("storage/bsreward-.png")

