# Predator-Prey Grid World MDP Evaluation

This part of the project implements a Reinforcement Learning framework to model an optimal predator strategy in an $N \times N$ grid world. It evaluates the expected discounted rewards (State Values) across varying grid sizes.

## 📂 Project Structure

Each core component is isolated in its own Python file to maintain modularity:

| File | Description |
| :--- | :--- |
| `simulator.py` | Handles the movement logic for both agents and reward triggers. |
| `kernel.py` | Generates the $\|S\|\|A\| \times \|S\|$ state-transition kernel. |
| `reward.py` | Generates the $\|S\| \times \|A\|$ expected reward matrix. |
| `sample_policy.py` | Implements the 50/50 distance-minimizing stochastic policy. |
| `induced_kernel.py` | Computes the policy-induced transition matrix $P^\pi$. |
| `induced_reward.py` | Computes the policy-induced reward vector $R^\pi$. |
| `state_value_eval.py` | Iteratively solves the Bellman Equation for $V^\pi$. |
| `q_value_eval.py` | Derives action-values $Q^\pi(s, a)$ from state-values. |
| **`Q1j.ipynb`** | **Main entry point.** Orchestrates simulation and plotting. |

## 🛠 Prerequisites

Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib scipy
```

## 🚀 How to Run

1. **Setup Directory:** Ensure all `.py` files are in the parent folder of the `Q1` folder, where `Q1j.ipynb` is located.
2. **Launch Notebook:** Open `Q1j.ipynb` using Jupyter Lab, VS Code, or Jupyter Notebook.
3. **Execute Cells in Sequence:**
    - **Cell 1 (Imports):** Pulls the logic from the separate `.py` modules.
    - **Cell 2 (Simulation):** Calls `run_evaluation()` which iterates through $N \in \{4,6,8,10,12\}$.
    - **Cell 3 (Plotting):** Generates the "State Value vs N" and "Run Time vs N" graphs.

## 📊 Analysis

- **Initial State Value:** Evaluates $V^\pi((1,1), (N,N))$ for the specified policy. As $N$ increases, you should observe a decay in value due to the $\gamma = 0.99$ discount factor over longer Manhattan distances.
- **Run Time:** Monitors performance. Note that the state space grows at $O(N^4)$, so computation time increases significantly for $N=12$.