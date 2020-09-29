"""
    Environment Repository: https://github.com/michaelliyunhao/RL-project
    Installation Manual: https://git.ias.informatik.tu-darmstadt.de/quanser/clients
"""

import gym
import quanser_robots

env = gym.make('Qube-v0')
env.reset()
env.render()
