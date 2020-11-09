# 2020 EI339
EI339 Artificial Intelligence, 2020 Fall, SJTU


<br>

### Sample File Tree
```
├─Project 1-1
│  │  constants.py
│  │  environment.py
│  │  evaluate.py
│  │  policy.py
│  │  train.py
│  │  update.py
│  │  
│  ├─assist
│  │      sxsybxtyl.py
│  │      
│  ├─utilities
│  │  ├─Policy_Iteration
│  │  │      train__search_param.py
│  │  │      
│  │  └─Q_Learning
│  │          train__eval_param.py
│  │          train__search_param.py
│  │          
│  ├─_trained
│  │  ├─Policy_Iteration
│  │  │      TestOutput.npy
│  │  │      
│  │  └─Q_Learning
│  │      │  epoch=10000_lr=0.0250_df=0.0250_ep=0.0250.npy
│  │      │  TestOutput.npy
│  │      │  
│  │      └─eval
│  │              evaluation_report_20201009144930
│  │              evaluation_report_20201009145027
│  │              evaluation_report_20201009174536
│  │              
│  └─__pycache__
│          constants.cpython-37.pyc
│          environment.cpython-37.pyc
│          policy.cpython-37.pyc
│          update.cpython-37.pyc
│   
├─Project 1-2
│  │  _env_install_test.py      Test environment and library
│  │  
│  ├─MPC
│  │  │  README.md
│  │  │  
│  │  ├─MPC-BallBalancer
│  │  │  │  config.yml          Parameter setting file
│  │  │  │  controller.py       Model predict controller
│  │  │  │  dynamics.py         Build dynamics functions
│  │  │  │  run.py              Training precess 
│  │  │  │  utils.py            Environment analysis
│  │  │  │  
│  │  │  └─Hive                 Optimization algorithm
│  │  │          Hive.py
│  │  │          README.md
│  │  │          SelectionMethods.py
│  │  │          Utilities.py
│  │  │          __init__.py
│  │  │          
│  │  ├─MPC-CartPoleSwing
│  │  │  │  config.yml          Parameter setting file
│  │  │  │  controller.py       Model predict controller
│  │  │  │  dynamics.py         Build dynamics functions
│  │  │  │  run.py              Training precess 
│  │  │  │  utils.py            Environment analysis
│  │  │  │  
│  │  │  ├─Hive                 Optimization algorithm
│  │  │  │  │  Hive.py
│  │  │  │  │  README.md
│  │  │  │  │  SelectionMethods.py
│  │  │  │  │  Utilities.py
│  │  │  │  │  __init__.py
│  │  │  │  │  
│  │  │  │  └─__pycache__
│  │  │  │          Hive.cpython-37.pyc
│  │  │  │          Utilities.cpython-37.pyc
│  │  │  │          __init__.cpython-37.pyc
│  │  │  │          
│  │  │  ├─storage              Store model and figure
│  │  │  │      config-1.yml
│  │  │  │      config-2.yml
│  │  │  │      config-3.yml
│  │  │  │      config-4.yml
│  │  │  │      config-5.yml
│  │  │  │      data_exp_1.pkl
│  │  │  │      exp_1.ckpt
│  │  │  │      loss-1.png
│  │  │  │      loss-2.png
│  │  │  │      model_error_exp_1.png
│  │  │  │      model_error_exp_2.png
│  │  │  │      reward-1.png
│  │  │  │      reward-2.png
│  │  │  │      
│  │  │  └─__pycache__
│  │  │          controller.cpython-37.pyc
│  │  │          dynamics.cpython-37.pyc
│  │  │          utils.cpython-37.pyc
│  │  │          
│  │  └─MPC-Qube
│  │      │  config.yml         Parameter setting file
│  │      │  controller.py      Model predict controller
│  │      │  dynamics.py        Build dynamics functions
│  │      │  run.py             Training precess
│  │      │  utils.py           Environment analysis
│  │      │  
│  │      ├─Hive                Optimization algorithm
│  │      │  │  Hive.py
│  │      │  │  README.md
│  │      │  │  SelectionMethods.py
│  │      │  │  Utilities.py
│  │      │  │  __init__.py
│  │      │  │  
│  │      │  └─__pycache__
│  │      │          Hive.cpython-37.pyc
│  │      │          Utilities.cpython-37.pyc
│  │      │          __init__.cpython-37.pyc
│  │      │          
│  │      ├─storage             Store model and figure
│  │      │      config-1.yml
│  │      │      config-2.yml
│  │      │      config-4.yml
│  │      │      config-5.yml
│  │      │      config-6.yml
│  │      │      config-7.yml
│  │      │      config_3.yml
│  │      │      data_exp_7.pkl
│  │      │      exp_7.ckpt
│  │      │      loss-1.png
│  │      │      loss-2.png
│  │      │      loss-3.png
│  │      │      loss-4.png
│  │      │      loss-5.png
│  │      │      loss-6.png
│  │      │      loss-7.png
│  │      │      mpc.png
│  │      │      reward-1.png
│  │      │      reward-2.png
│  │      │      reward-3.png
│  │      │      reward-4.png
│  │      │      reward-5.png
│  │      │      reward-6.png
│  │      │      reward-7.png
│  │      │      State Error h_0 100.png
│  │      │      
│  │      └─__pycache__
│  │              controller.cpython-37.pyc
│  │              dynamics.cpython-37.pyc
│  │              utils.cpython-37.pyc
│  │              
│  ├─TRPO
│  │      conjugate_gradients.py
│  │      LICENSE.md                Code license
│  │      main.py                   Training process
│  │      models.py                 Policy precidtion model
│  │      replay_memory.py          Store trajectories data
│  │      running_state.py          Adjust input data
│  │      trpo.py                   Optimization problem solver
│  │      utils.py                  Change the network parameter 
│  │      
│  ├─_env                           
│  │  │  _env_install_test.py
│  │  │  
│  │  └─quanser_robots              Quanser robot source code
│  │      │  .gitignore
│  │      │  Install.md
│  │      │  Readme.md
│  │      │  setup.py
│  │      │  unify_api.py
│  │      │  
│  │      ├─build
│  │      │  ├─bdist.win-amd64
│  │      │  └─lib
│  │      │      └─quanser_robots
│  │      │          │  common.py
│  │      │          │  __init__.py
│  │      │          │  
│  │      │          ├─ball_balancer
│  │      │          │      ball_balancer_rr.py
│  │      │          │      ball_balancer_sim.py
│  │      │          │      base.py
│  │      │          │      ctrl.py
│  │      │          │      __init__.py
│  │      │          │      
│  │      │          ├─cartpole
│  │      │          │      base.py
│  │      │          │      cartpole.py
│  │      │          │      cartpole_rr.py
│  │      │          │      ctrl.py
│  │      │          │      __init__.py
│  │      │          │      
│  │      │          ├─double_pendulum
│  │      │          │      base.py
│  │      │          │      ctrl.py
│  │      │          │      double_pendulum.py
│  │      │          │      double_pendulum_rr.py
│  │      │          │      __init__.py
│  │      │          │      
│  │      │          ├─levitation
│  │      │          │      base.py
│  │      │          │      ctrl.py
│  │      │          │      levitation.py
│  │      │          │      levitation_rr.py
│  │      │          │      __init__.py
│  │      │          │      
│  │      │          └─qube
│  │      │                  base.py
│  │      │                  ctrl.py
│  │      │                  qube.py
│  │      │                  qube_rr.py
│  │      │                  __init__.py
│  │      │                  
│  │      ├─quanser_robots
│  │      │  │  common.py
│  │      │  │  Readme.md
│  │      │  │  __init__.py
│  │      │  │  
│  │      │  ├─ball_balancer
│  │      │  │  │  ball_balancer_rr.py
│  │      │  │  │  ball_balancer_sim.py
│  │      │  │  │  base.py
│  │      │  │  │  ctrl.py
│  │      │  │  │  modeling_bb.pdf
│  │      │  │  │  modeling_rbu.pdf
│  │      │  │  │  Readme.md
│  │      │  │  │  __init__.py
│  │      │  │  │  
│  │      │  │  ├─examples
│  │      │  │  │      dummy_sim.py
│  │      │  │  │      pd_ctrl.py
│  │      │  │  │      pd_ctrl_rr.py
│  │      │  │  │      
│  │      │  │  └─__pycache__
│  │      │  │          __init__.cpython-37.pyc
│  │      │  │          
│  │      │  ├─cartpole
│  │      │  │  │  base.py
│  │      │  │  │  cartpole.py
│  │      │  │  │  cartpole_rr.py
│  │      │  │  │  ctrl.py
│  │      │  │  │  Readme.md
│  │      │  │  │  __init__.py
│  │      │  │  │  
│  │      │  │  ├─documentation
│  │      │  │  │      cartpole.jpg
│  │      │  │  │      model.pdf
│  │      │  │  │      
│  │      │  │  ├─examples
│  │      │  │  │      metronom.py
│  │      │  │  │      plotting_comparison.py
│  │      │  │  │      realData_230419.npz
│  │      │  │  │      simData_2304019.npz
│  │      │  │  │      swingup.py
│  │      │  │  │      
│  │      │  │  └─__pycache__
│  │      │  │          base.cpython-37.pyc
│  │      │  │          cartpole.cpython-37.pyc
│  │      │  │          __init__.cpython-37.pyc
│  │      │  │          
│  │      │  ├─double_pendulum
│  │      │  │  │  base.py
│  │      │  │  │  ctrl.py
│  │      │  │  │  double_pendulum.py
│  │      │  │  │  double_pendulum_rr.py
│  │      │  │  │  Readme.md
│  │      │  │  │  __init__.py
│  │      │  │  │  
│  │      │  │  ├─documentation
│  │      │  │  │      model.pdf
│  │      │  │  │      
│  │      │  │  ├─examples
│  │      │  │  │      balance.py
│  │      │  │  │      metronom.py
│  │      │  │  │      
│  │      │  │  └─__pycache__
│  │      │  │          __init__.cpython-37.pyc
│  │      │  │          
│  │      │  ├─levitation
│  │      │  │  │  base.py
│  │      │  │  │  ctrl.py
│  │      │  │  │  levitation.py
│  │      │  │  │  levitation_rr.py
│  │      │  │  │  model.pdf
│  │      │  │  │  Readme.md
│  │      │  │  │  __init__.py
│  │      │  │  │  
│  │      │  │  ├─examples
│  │      │  │  │      clsc_ctl.py
│  │      │  │  │      test.py
│  │      │  │  │      
│  │      │  │  └─__pycache__
│  │      │  │          __init__.cpython-37.pyc
│  │      │  │          
│  │      │  ├─qube
│  │      │  │  │  base.py
│  │      │  │  │  ctrl.py
│  │      │  │  │  model.pdf
│  │      │  │  │  qube.py
│  │      │  │  │  qube_rr.py
│  │      │  │  │  Readme.md
│  │      │  │  │  __init__.py
│  │      │  │  │  
│  │      │  │  ├─examples
│  │      │  │  │      metronome.py
│  │      │  │  │      param_env.py
│  │      │  │  │      swing-up.py
│  │      │  │  │      swing-up_rr.py
│  │      │  │  │      
│  │      │  │  └─__pycache__
│  │      │  │          base.cpython-37.pyc
│  │      │  │          ctrl.cpython-37.pyc
│  │      │  │          qube.cpython-37.pyc
│  │      │  │          __init__.cpython-37.pyc
│  │      │  │          
│  │      │  └─__pycache__
│  │      │          common.cpython-37.pyc
│  │      │          __init__.cpython-37.pyc
│  │      │          
│  │      └─quanser_robots.egg-info
│  │              dependency_links.txt
│  │              not-zip-safe
│  │              PKG-INFO
│  │              requires.txt
│  │              SOURCES.txt
│  │              top_level.txt
│  │              
│  └─_reference_RL-project_michaelliyunhao      Reference code
│      │  README.md
│      │  
│      ├─DQN
│      │  │  README.md
│      │  │  
│      │  ├─DQN-CartPoleStab
│      │  │  │  config.yml
│      │  │  │  DQN.py
│      │  │  │  README.md
│      │  │  │  test.py
│      │  │  │  test_rr.py
│      │  │  │  train.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  └─storage
│      │  │          config-1.yml
│      │  │          config-2.yml
│      │  │          config-3.yml
│      │  │          config-4.yml
│      │  │          exp_4.ckpt
│      │  │          loss-1.png
│      │  │          loss-2.png
│      │  │          loss-3.png
│      │  │          loss-4.png
│      │  │          loss-5.png
│      │  │          loss-6.png
│      │  │          README.md
│      │  │          reward-1.png
│      │  │          reward-2.png
│      │  │          reward-3.png
│      │  │          reward-4.png
│      │  │          reward-5.png
│      │  │          reward-6.png
│      │  │          
│      │  ├─DQN-Double
│      │  │  │  config.yml
│      │  │  │  DQN.py
│      │  │  │  README.md
│      │  │  │  test.py
│      │  │  │  train.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  └─storage
│      │  │          config-0.yml
│      │  │          config-1.yml
│      │  │          config-10.yml
│      │  │          config-11.yml
│      │  │          config-12.yml
│      │  │          config-13.yml
│      │  │          config-2.yml
│      │  │          config-3.yml
│      │  │          config-4.yml
│      │  │          config-5.yml
│      │  │          config-6.yml
│      │  │          config-7.yml
│      │  │          config-8.yml
│      │  │          config-9.yml
│      │  │          exp_4.ckpt
│      │  │          loss-0.png
│      │  │          loss-1.png
│      │  │          loss-10.png
│      │  │          loss-11.png
│      │  │          loss-12.png
│      │  │          loss-13.png
│      │  │          loss-2.png
│      │  │          loss-3.png
│      │  │          loss-4.png
│      │  │          loss-5.png
│      │  │          loss-6.png
│      │  │          loss-7.png
│      │  │          loss-8.png
│      │  │          loss-9.png
│      │  │          README.md
│      │  │          reward-0.png
│      │  │          reward-1.png
│      │  │          reward-10.png
│      │  │          reward-11.png
│      │  │          reward-12.png
│      │  │          reward-13.png
│      │  │          reward-2.png
│      │  │          reward-3.png
│      │  │          reward-4.png
│      │  │          reward-5.png
│      │  │          reward-6.png
│      │  │          reward-7.png
│      │  │          reward-8.png
│      │  │          reward-9.png
│      │  │          
│      │  ├─DQN-Qube
│      │  │  │  config.yml
│      │  │  │  DQN.py
│      │  │  │  README.md
│      │  │  │  test.py
│      │  │  │  test_on_real_platform.py
│      │  │  │  train.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  └─storage
│      │  │          .~lock.Parameters.ods#
│      │  │          config-1.yml
│      │  │          config-2.yml
│      │  │          config-3.yml
│      │  │          config-5.yml
│      │  │          config-6.yml
│      │  │          config-7.yml
│      │  │          config-8.yml
│      │  │          config-9.yml
│      │  │          data_real_world.pkl
│      │  │          exp_6.ckpt
│      │  │          loss-1.png
│      │  │          loss-2.png
│      │  │          loss-3.png
│      │  │          loss-4.png
│      │  │          loss-5.png
│      │  │          loss-6.png
│      │  │          loss-7.png
│      │  │          loss-8.png
│      │  │          loss-9.png
│      │  │          Parameters.ods
│      │  │          README.md
│      │  │          reward-1.png
│      │  │          reward-2.png
│      │  │          reward-3.png
│      │  │          reward-4.png
│      │  │          reward-5.png
│      │  │          reward-6-real-world.png
│      │  │          reward-6.png
│      │  │          reward-7.png
│      │  │          reward-8.png
│      │  │          reward-9.png
│      │  │          simulatedModelOnRealPlatform-2.png
│      │  │          simulatedModelOnRealPlatform-3.png
│      │  │          simulatedModelOnRealPlatform-4.png
│      │  │          simulatedModelOnRealPlatform.png
│      │  │          
│      │  └─DQN-Swing
│      │      │  config.yml
│      │      │  DQN.py
│      │      │  README.md
│      │      │  test.py
│      │      │  test_rr.py
│      │      │  train.py
│      │      │  utils.py
│      │      │  
│      │      └─storage
│      │              config-0.yml
│      │              config-1.yml
│      │              config-2.yml
│      │              config-3.yml
│      │              exp_0.ckpt
│      │              exp_1_best.ckpt
│      │              loss-0.png
│      │              loss-1-find-best.png
│      │              loss-1.png
│      │              loss-2.png
│      │              loss-3.png
│      │              README.md
│      │              reward-0.png
│      │              reward-1-find-best.png
│      │              reward-1.png
│      │              reward-2.png
│      │              reward-3.png
│      │              
│      ├─MPC
│      │  │  README.md
│      │  │  
│      │  ├─MPC-CartPoleStab
│      │  │  │  config.yml
│      │  │  │  controller.py
│      │  │  │  dynamics.py
│      │  │  │  example.ipynb
│      │  │  │  README.md
│      │  │  │  run.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  ├─.idea
│      │  │  │      misc.xml
│      │  │  │      modules.xml
│      │  │  │      MPC qube.iml
│      │  │  │      workspace.xml
│      │  │  │      
│      │  │  ├─.ipynb_checkpoints
│      │  │  │      example-checkpoint.ipynb
│      │  │  │      
│      │  │  ├─Hive
│      │  │  │      Hive.py
│      │  │  │      README.md
│      │  │  │      SelectionMethods.py
│      │  │  │      Utilities.py
│      │  │  │      __init__.py
│      │  │  │      
│      │  │  └─storage
│      │  │          config-1.yml
│      │  │          config-2.yml
│      │  │          exp_1.ckpt
│      │  │          loss-1.png
│      │  │          loss-2.png
│      │  │          model_error_exp_1.png
│      │  │          reward-1.png
│      │  │          reward-2.png
│      │  │          
│      │  ├─MPC-CartPoleSwing
│      │  │  │  config.yml
│      │  │  │  controller.py
│      │  │  │  dynamics.py
│      │  │  │  README.md
│      │  │  │  run.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  ├─Hive
│      │  │  │      Hive.py
│      │  │  │      README.md
│      │  │  │      SelectionMethods.py
│      │  │  │      Utilities.py
│      │  │  │      __init__.py
│      │  │  │      
│      │  │  └─storage
│      │  │          config-1.yml
│      │  │          config-2.yml
│      │  │          config-3.yml
│      │  │          config-4.yml
│      │  │          config-5.yml
│      │  │          loss-1.png
│      │  │          loss-2.png
│      │  │          model_error_exp_1.png
│      │  │          model_error_exp_2.png
│      │  │          reward-1.png
│      │  │          reward-2.png
│      │  │          
│      │  ├─MPC-Double
│      │  │  │  config.yml
│      │  │  │  controller.py
│      │  │  │  dynamics.py
│      │  │  │  README.md
│      │  │  │  run.py
│      │  │  │  utils.py
│      │  │  │  
│      │  │  ├─Hive
│      │  │  │      Hive.py
│      │  │  │      README.md
│      │  │  │      SelectionMethods.py
│      │  │  │      Utilities.py
│      │  │  │      __init__.py
│      │  │  │      
│      │  │  └─storage
│      │  │          config-1.yml
│      │  │          config-2.yml
│      │  │          config-3.yml
│      │  │          config-4.yml
│      │  │          loss-1.png
│      │  │          loss-2.png
│      │  │          loss-3.png
│      │  │          loss-4.png
│      │  │          model_error_exp_1.png
│      │  │          reward-1.png
│      │  │          reward-2.png
│      │  │          reward-3.png
│      │  │          reward-4.png
│      │  │          
│      │  └─MPC-Qube
│      │      │  config.yml
│      │      │  controller.py
│      │      │  dynamics.py
│      │      │  README.md
│      │      │  run.py
│      │      │  test.py
│      │      │  utils.py
│      │      │  
│      │      ├─Hive
│      │      │  │  Hive.py
│      │      │  │  README.md
│      │      │  │  SelectionMethods.py
│      │      │  │  Utilities.py
│      │      │  │  __init__.py
│      │      │  │  
│      │      │  └─__pycache__
│      │      │          Hive.cpython-37.pyc
│      │      │          Utilities.cpython-37.pyc
│      │      │          __init__.cpython-37.pyc
│      │      │          
│      │      ├─storage
│      │      │      Angle Error h_0 100.png
│      │      │      config-1.yml
│      │      │      config-2.yml
│      │      │      config-4.yml
│      │      │      config-5.yml
│      │      │      config-6.yml
│      │      │      config-7.yml
│      │      │      config_3.yml
│      │      │      data_exp_7.pkl
│      │      │      exp_7.ckpt
│      │      │      loss-1.png
│      │      │      loss-2.png
│      │      │      loss-3.png
│      │      │      loss-4.png
│      │      │      loss-5.png
│      │      │      loss-6.png
│      │      │      loss-7.png
│      │      │      mpc.png
│      │      │      reward-1.png
│      │      │      reward-2.png
│      │      │      reward-3.png
│      │      │      reward-4.png
│      │      │      reward-5.png
│      │      │      reward-6.png
│      │      │      reward-7.png
│      │      │      State Error h_0 100.png
│      │      │      
│      │      └─__pycache__
│      │              controller.cpython-37.pyc
│      │              dynamics.cpython-37.pyc
│      │              utils.cpython-37.pyc
│      │              
│      └─Resources
│          │  README.md
│          │  
│          ├─DQN
│          │      Playing Atari with Deep Reinforcement Learning.pdf
│          │      Q-Learning in Continuous State Action Spaces.pdf
│          │      README.md
│          │      
│          ├─figures
│          │      qube-after-fine-tuning.gif
│          │      qube-before-fine-tuning.gif
│          │      qube.gif
│          │      README.md
│          │      stabe.gif
│          │      swing.gif
│          │      swing_interesting.gif
│          │      
│          └─MPC
│                  Approximate Dynamic Programming with Gaussian Processes.pdf
│                  Constrained model predictive control_ Stability and optimality.pdf
│                  Neural Network Dynamics for Model based Deep Rl with Model free fine tuning.pdf
│                  README.md
│                  [edit] Neural Network Dynamics for Model based Deep Rl with Model free fine tuning.pdf
│                  
└─_test
        EI338_Hw3_prob1.py
```

# Project 1
## Easy 21

## Robot
