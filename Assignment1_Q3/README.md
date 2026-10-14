# Predator-Prey Model Estimation and Robustness Analysis (Q3)

This part of the project extends the Reinforcement Learning framework to scenarios where the exact transition probabilities of the environment are unknown. It focuses on estimating the state-transition kernel through sampling from a simulator and analyzing how the approximation error behaves as the number of samples ($K$) increases.

## 📂 Project Structure

The workflow utilizes the optimized solvers developed in Question 2 alongside the new estimation logic required for Question 3:

| File | Description |
| :--- | :--- |
| `simulator.py` | The environment simulator that models predator and prey movement logic. |
| `kernel.py` | Provides the exact transition kernel used as the "Gold Standard" for comparison. |
| `reward.py` | Generates the $N^2 \times N^2$ reward matrix for the grid world. |
| `value_iteration.py` | Optimized Bellman solver used to derive $Q$-functions from kernels. |
| **`estimate_kernel.py`** | Implements frequency-based kernel estimation by sampling the simulator $K$ times per state-action pair. |
| **`Q3bc.ipynb`** | **Main entry point.** Orchestrates the estimation trials, calculates $L_1$ errors, and performs statistical analysis over multiple seeds. |

## 🛠 Prerequisites

Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib scipy
```
## 🚀 How to Run

1. **Environment Setup:** Ensure all `.py` files are in the parent folder of the `Q3` folder, where `Q3bc.ipynb` is located.
2. **Launch Notebook:** Open your notebook file in Jupyter Lab or VS Code.
3. **Execute Analysis:**
   - **Part (b):** Run the cells to compare the estimated $\hat{Q}$ against the exact $Q^*$ for $N=5$ across $K \in \{5, 10, 15, 20, 25\}$.
   - **Part (c):** Run the statistical section to repeat the process over 5 different random seeds to generate the Mean and Standard Deviation plots.

## 📊 Analysis

* **Accuracy vs. Samples ($K$):** As the number of samples per state-action pair increases, the estimated kernel converges to the true transition probabilities. The $L_1$ error decreases as the model becomes more representative of the true environment.
* **Statistical Stability:** By evaluating 5 different seeds, the project measures the estimator's sensitivity to the stochastic movement of the prey. The standard deviation (shaded region in the plots) narrows as $K$ increases, indicating a more robust estimate.
* **Complexity:** For $N=5$ and $K=25$, the simulator is called $|\mathcal{S}||\mathcal{A}|K = 78,125$ times per trial. The implementation utilizes sparse matrix structures to handle this volume without memory overflow.