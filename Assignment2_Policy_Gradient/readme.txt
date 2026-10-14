================================================================================
README: Policy Gradient Training (SGA vs. Adam)
================================================================================

This repository contains the code to train a policy network using REINFORCE 
(Vanilla Policy Gradient). It compares standard Stochastic Gradient Ascent (SGA) 
against the Adam optimizer.

--------------------------------------------------------------------------------
1. DIRECTORY STRUCTURE & REQUIREMENTS
--------------------------------------------------------------------------------
CRITICAL: Before running the code, ensure that ALL of the following files are 
placed together in the exact same directory (the `Policy_Gradient` folder). 
The Jupyter notebook relies on these Python scripts being in the same folder 
to import the necessary classes and functions.

Files required in the directory:
- Q1cd.ipynb                   (Main Jupyter Notebook)
- adam_train.py                (Adam optimizer training loop)
- gradient_estimate.py         (Monte Carlo policy gradient estimator)
- policy_network.py            (Neural network architecture)
- rollout_discounted_reward.py (Helper function for evaluation rollouts)
- simple_SGA.py                (Vanilla SGA training loop)
- simulator.py                 (Environment simulator)

Prerequisites:
- Python 3.x
- Jupyter Notebook / JupyterLab
- PyTorch (torch)
- Matplotlib (matplotlib.pyplot)
- NumPy (numpy)

--------------------------------------------------------------------------------
2. HOW TO RUN THE CODE
--------------------------------------------------------------------------------
To execute the training and generate the learning curves, follow these steps:

Step 1: Open the Notebook
Launch Jupyter Notebook or JupyterLab and open the `Q1cd.ipynb` file.

Step 2: Run Cell 1 (Setup and Imports)
Execute the first cell. This cell will import all the required external libraries 
(torch, matplotlib, numpy, etc.) as well as the custom classes and methods from 
the Python scripts listed above. 

Step 3: Run Cell 2 (Train with Simple SGA)
Execute the second cell. This cell will initialize the default policy network, 
train it using the `simple_SGA` function across different learning rates, and 
output the moving-average learning curve plot for the SGA optimizer.

Step 4: Run Cell 3 (Train with Adam)
Execute the third cell. This cell will initialize a fresh policy network, train 
it using the `adam_train` function across different learning rates, and output 
the moving-average learning curve plot for the Adam optimizer.

Note: The training cells may take a few minutes to complete depending on your 
hardware, as they run multiple iterations of Monte Carlo rollouts. Status 
updates will be printed to the console during training.