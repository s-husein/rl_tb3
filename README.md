# Exploration of an Unknown Environment using Deep Reinforcement Learning and Intrinsic Rewards
This repository contains a ROS (Robot Operating Sytem) package for created for my Final Year Project (FYP). This project was for the completion of my BS Electronic Engineering Batch Fall-2020 Degree at Balochistan University of Information Technology, Engineering and Management Sciences (BUITEMS) Quetta.


The main target in this project was to train a neural network that can guide a mobile robot to explore its surrounding envrironment while only using the depth information from the environment using a depth camera. The prototype mobile robot used in this project is a Turtlebot3 burger that has an Intel RealSense D455 depth camera mounted on it. 

The package contains:
* Modified URDF and gazebo files for the turtlebot3 burger that includes the depth camera as well. The files are located in the `urdf` folder.
* Simulation worlds such mazes and obstacle courses for training of the RL agent. The world files are in the `worlds` folder.
* Open AI gym environment that runs as a node and you can get RGB and depth images from the gazebo simulation environment and send continuous or discrete actions to the robot. This can be used both for simulation and hardware implementation. Location: `scripts/gymenv.py`.
* Launch files for initiating the gazebo simulation environment, launching of URDF and mapping files (work for both hardware and software), the launch files are in the `launch` folder.
* Scripts for automating the creation of neural networks (only CNN and MLP can be created). The file is `scripts/net.py`.
* Algorithms and training scripts for the algorithms. There are multiple files located in the `scripts` folder whereas the `algos.py` contains algorithms such as PPO, A2C and REINFORCE and also contains the implementation of RND (Random Network Distillation). The code is inspired by `stablebaslines3`
