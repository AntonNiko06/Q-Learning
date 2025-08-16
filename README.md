# Q-Learning from scratch
Implementation of reinforcement learning agent using Q-Learning for navigating a simple Gridworld / Frozen Lake environment, built without the use of external RL libraries.

## Aim
The aim of this project is to get my foot in the door of reinforcement learning by building a simple project from scratch, without the use of any RL libraries. More specifically, I wanted to understand what the learning process of agents in a Gridworld environment looks like.

This fits into my broader learning approach for the field of Machine Learning, which is to:
1. Build several models completely from scratch to deeply understand the theory and math underlying them
2. Build on that knowledge by creating more complex projects where the focus shifts from raw implementation and theory to practical results for real-world problems.

## Demo
### Agent at the start
![Agent at the start](assets/traversal_start.gif)
### Agent at the end
![Agent at the end](assets/traversal_end.gif)

## Features
- Implementation of Q-Learning algorithm from scratch
- Adjustable hyperparameters and maze layout
- Visualization of agent's traversal through maze

## How It Works
The agent moves through the maze either randomly or by performing the action he assigns the highest value to. After each step, the value of performing that action on that field is updated using the Q-Function.

## Google Colab
You can run this project using Google Colab instead of your local machine (Note that Pygame visualisations are not supported by Google Colab): 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AntonNiko06/Q-Learning/blob/main/q_learning.ipynb)


## Local Installation
[[ guide to installation ]]

### Usage
[[ how to run ]]

## Possible Improvements
[[ what could be done to improve performance? Speed up training? etc. ]]
