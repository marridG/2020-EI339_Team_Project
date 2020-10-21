import numpy as np
from Hive import Hive
from Hive import Utilities

class MPC(object):
    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]
        self.evaluator = Evaluator(self.gamma)

    def act(self, state, dynamic_model):
        """
        Optimize the action by Artificial Bee Colony algorithm
        :param state: (numpy array) current state
        :param dynamic_model: system dynamic model
        :return: (float) optimal action
        """
        self.evaluator.update(state, dynamic_model)
        optimizer = Hive.BeeHive(lower=[float(self.action_low)] * self.horizon,
                                 upper=[float(self.action_high)] * self.horizon,
                                 fun=self.evaluator.evaluate,
                                 numb_bees=self.numb_bees,
                                 max_itrs=self.max_itrs,
                                 verbose=False)
        cost = optimizer.run()
        # print("Solution: ",optimizer.solution[0])
        # print("Fitness Value ABC: {0}".format(optimizer.best))
        # Uncomment this if you want to see the performance of the optimizer
        # Utilities.ConvergencePlot(cost)
        return optimizer.solution[0]

    def get_action(self, state, dynamic_model):
        self.evaluator.update(state, dynamic_model)
        obs = np.array([state for _ in range(self.max_itrs)])
        trajectory_cost_list = np.zeros(self.max_itrs)
        obs_next_list, act_list = [], []
        for _ in range(self.horizon):  # tqdm.tqdm(range(self.horizon)):
            actions = np.array([self.env.action_space.sample() for _ in range(self.max_itrs)])
            act_list.append(actions)
            input_data = np.column_stack((obs, actions))
            _actions_rewards = self.evaluator.get_cost_array(obs, actions)
            trajectory_cost_list += _actions_rewards
            obs = dynamic_model.predict(input_data)
            obs_next_list.append(obs)
        j = np.argmin(trajectory_cost_list)
        return act_list[0][j]  # 返回一个数组

class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions): #这里的action只有一维了
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate((state_tmp, actions[j]))
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    # need to change this function according to different environment
    def get_reward(self, obs, action_n):
        Q = np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2])  # see dim of state space
        R = np.diag([1e-4, 1e-4])  # see dim of action space
        _state_des = np.zeros(8,)

        err_s = (_state_des - obs).reshape(-1,)  # or self._state
        err_a = action_n.reshape(-1,)
        quadr_cost = err_s.dot(Q.dot(err_s)) + err_a.dot(R.dot(err_a))

        obs_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, 4.*np.pi, 4.*np.pi, 0.5, 0.5])
        act_max = np.array([5.0, 5.0])
        max_cost = obs_max.dot(Q.dot(obs_max)) + act_max.dot(R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        c_max = -1.0 * np.log(1e-4) / max_cost

        # Calculate the scaled exponential
        rew = np.exp(-c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
        return float(rew)

    @staticmethod
    def get_cost_array(obs, action_n):
        Q = np.array([np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2]) for _ in range(10)]) # see dim of state space
        R = np.array([np.diag([1e-4, 1e-4]) for _ in range(10)])  # see dim of action space
        _state_des = np.zeros(8,)
        # action_n(10, 2)
        # Q(10, 8, 8)
        # error_s(10, 8, 1)
        # error_a(10, 2, 1)
        err_s = (_state_des - obs)[:,:,np.newaxis] # or self._state
        err_a = action_n[:,:,np.newaxis]
        tmp1 = np.squeeze(np.matmul(Q, err_s),2) #(10,8)
        tmp2 = np.squeeze(np.matmul(R, err_a), 2) #(10,2)
        # print("dot1",np.squeeze(err_s,2).dot(tmp).shape)
        quadr_cost = np.multiply(np.squeeze(err_s,2),tmp1).sum(-1) + np.multiply(np.squeeze(err_a,2),tmp2).sum(-1)
        return quadr_cost #(10,1)