# Humanoid robot playing balance board with PPO (NVIDIA Isaac Gym)
This repository contains the code and configuration files for humanoid robot playing balance board in the NVIDIA Isaac Gym simulator. This work was done as part of the paper titled "Reinforcement Learning and Action Space Shaping for a Humanoid Agent in a Highly Dynamic Environment."

Demo:

![Demo](Docs/Demo.gif)

## Training Details

The agent aims to remain stable on the balance board while avoiding letting itself or the board touch the gorund. We designed different control schemes for our agent and compare their performance. Therefore, the observation space and the action space will be different according to the schemes we applied. 

The reward function is conditioned on board’s angle to the ground, the orthogonal projection of the agent’s orientation to the z-axis (upwards direction), the angular velocity of the board, and the distance between the agent’s left foot and an optimal position on top of the board. 

Observation Space:
Feature | Dimensionality |
--- | --- | 
Position of Controlled Joints | [6-23]* | 
Velocity of Controlled Joints | [6-23]* |
Previous Actions              | [6-23]* |
Agent’s Position | 3
Agent’s Angular Velocity | 3
Agent’s Orientation | 3
Left and Right foot position (Relative to CoM) | 6
Distance of Left Foot to Ideal Position | 1
Upwards Projection of Robot Torso | 1
Board Position | 3
Board Angle | 1
Roller Position | 3

Action Space was the number of the joints the agent can control.

The agent was trained with PPO using the NVIDIA Isaac Gym simulator. Training for 300 million timesteps which takes under 40 minutes of real-time on a single RTX 2080 Super.

For more information, you can check our paper under `Docs/`.

### Files

`Code/tasks/fixed_robinion_balance_board_sim.py` contains the implementation of the environment. This file should be placed under the `isaacgymenvs/tasks/` folder in the Isaac Gym simulator.

`Code/cfg/task/FixedRobinionBalanceBoardSim.yaml` contains the parameters for the environment. This file should be placed under `isaacgymenvs/cfg/task`.

`Code/cfg/train/FixedRobinionBalanceBoardSimPPO.yaml` contains the parameters for the RL algorithm. This file should be placed under `isaacgymenvs/cfg/train`.

`Assets/` contains the robot and balance board models and their URDF description.
