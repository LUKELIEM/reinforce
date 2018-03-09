# Policy-based Reinforcement Learning using PyTorch

## Overview

In this folder, I have a series of Jupyter Notebooks documenting my learning of policy-based reinforcement learning. They implement:

* Andrej Karpathy's REINFORCE agent written in Python and numpy for Pong (Atari)   
* PyTorch's example codes for REINFORCE and Actor-Critic for Cartpole  
* My REINFORCE and Actor-Critic agents written in PyTorch for Pong  

I discovered that getting RL models to work is a lot more difficult than getting convolutional neural networks to perform supervised learning tasks such as image classification:  

* The hyperparameters is different. There is learning rate, but no regularization.  
* The training workflow is very different. Instead of performing a forward pass on the data followed by a backward pass as in image classification, policy-based RL involves running the forward pass hundreds to thousands of games steps in an episode (depending on the game), accumulating a stack of rewards and log-probability along the way, then using these accumulated data to perform a single backward pass to update the policy parameters.  
* Policy-based RL is gradient ascent instead of gradient descent.  
* There are many local optima in the policy landscape and the agent can easily get trapped in these local optima. They easily get content being "King of the Small Hill".  

## Environment

You need to install the following to run these notebooks:
1. PyTorch with Python 3.6.X
2. OpenAI Gym

## Content

A general description of the Notebooks under that folder:

1. [pg-cartpole.ipynb](./pg-cartpole.ipynb)    
This Notebook implements PyTorch's REINFORCE and ACTOR-CRITIC agents for OpenAI's Cartpole. It demonstrates the superiority of Actor-Critic over REINFORCE.

2. [karpathy-pong.ipynb](./karpathy-pong.ipynb)    
This Notebook trains Karpathy's numpy implementation of REINFORCE on PONG to 30K episodes, when it achieves over human performance.

3. [reinforce.ipynb](./reinforce.ipynb)    
This Notebook adapted PyTorch's ACTOR_CRITIC agent for Pong and enables it to run with GPU acceleration.

4. gym/actor-critic.ipynb    
This Notebook adapted PyTorch's ACTOR_CRITIC agent for Pong and enables it to run with GPU acceleration.

5. gym/eval_model.ipynb    
This Notebook analyzes the content the model and history files, allowing users to generate images of the weight parameters and plots of scores versus training episodes.

6. gym/downsample.ipynb    
This Notebook look at different ways of downsampling PONG's images down to 40x40 or 80x80 frames.
