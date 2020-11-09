# 2020 EI339
EI339 Artificial Intelligence, 2020 Fall, SJTU


<br>

### Sample File Tree
```
~
├─Project 1-1                   HIGHLY WRAPPED IMPLEMENTATION - Problem 1: Q-learning, Policy Iteration
│  │  constants.py              Constants: Environment settings, training settings, etc.
│  │  environment.py            Easy21 Evironment implementation
│  │  evaluate.py               Evaluation environment implementation
│  │  policy.py                 Action policies classes
│  │  train.py                  Train model classes implementation
│  │  update.py                 Update functions classes implementation
│  │  
│  ├─assist
│  │      sxsybxtyl.py          Generate the transition probabilities
│  │      
│  ├─utilities                  Utilities
│  │  ├─Policy_Iteration
│  │  │      train__search_param.py     Try and train using different hyper-parameters
│  │  │      
│  │  └─Q_Learning
│  │          train__eval_param.py      Try and evaluate the trained models of different hyper-parameters
│  │          train__search_param.py    Try and train using different hyper-parameters
│  │          
│  └─_trained                           Trained models & evaluation output path
│     ├─Policy_Iteration
│     │      TestOutput.npy
│     │      
│     └─Q_Learning
│         │  TestOutput.npy
│         │  
│         └─eval
│                 evaluation_report_20201009144930
│          
└─Project 1-2
   │  _env_install_test.py      Test environment and library
   │  
   ├─MPC
   │  │  README.md
   │  │  
   │  ├─MPC-BallBalancer
   │  │  │  config.yml          Parameter setting file
   │  │  │  controller.py       Model predict controller
   │  │  │  dynamics.py         Build dynamics functions
   │  │  │  run.py              Training precess 
   │  │  │  utils.py            Environment analysis
   │  │  │  
   │  │  └─Hive                 Optimization algorithm
   │  │          Hive.py
   │  │          README.md
   │  │          SelectionMethods.py
   │  │          Utilities.py
   │  │          __init__.py
   │  │          
   │  ├─MPC-CartPoleSwing
   │  │  │  config.yml          Parameter setting file
   │  │  │  controller.py       Model predict controller
   │  │  │  dynamics.py         Build dynamics functions
   │  │  │  run.py              Training precess 
   │  │  │  utils.py            Environment analysis
   │  │  │  
   │  │  ├─Hive                 Optimization algorithm
   │  │  │  │  Hive.py
   │  │  │  │  README.md
   │  │  │  │  SelectionMethods.py
   │  │  │  │  Utilities.py
   │  │  │  └─ __init__.py
   │  │  │          
   │  │  └─storage              Store model and figure
   │  │          
   │  └─MPC-Qube
   │      │  config.yml         Parameter setting file
   │      │  controller.py      Model predict controller
   │      │  dynamics.py        Build dynamics functions
   │      │  run.py             Training precess
   │      │  utils.py           Environment analysis
   │      │  
   │      ├─Hive                Optimization algorithm
   │      │  │  Hive.py
   │      │  │  README.md
   │      │  │  SelectionMethods.py
   │      │  │  Utilities.py
   │      │  │  __init__.py
   │      │  │  
   │      │  └─__pycache__
   │      │          Hive.cpython-37.pyc
   │      │          Utilities.cpython-37.pyc
   │      │          __init__.cpython-37.pyc
   │      │          
   │      └─storage             Store model and figure
   │              
   └─TRPO
          conjugate_gradients.py
          LICENSE.md                Code license
          main.py                   Training process
          models.py                 Policy precidtion model
          replay_memory.py          Store trajectories data
          running_state.py          Adjust input data
          trpo.py                   Optimization problem solver
          utils.py                  Change the network parameter 

```

# Project 1
## Easy 21

## Robot
