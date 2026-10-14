# Predator-Prey Optimal Control: Value vs. Policy Iteration

This part of the project focuses on finding the optimal attacking strategy for a predator in an $N \times N$ grid world. It implements and compares two fundamental Dynamic Programming algorithms: **Value Iteration** and **Policy Iteration**.

## 📂 Project Structure

The project maintains a modular architecture. Q2 builds upon the environment dynamics established in Q1:

### Core Environment (from Q1)
| File | Description |
| :--- | :--- |
| `kernel.py` | Generates the state-transition kernel $P(s'\|s,a)$. |
| `reward.py` | Generates the expected reward matrix $R(s,a)$. |
| `state_value_eval.py` | Used by Policy Iteration for the evaluation phase. |

### Optimization Logic (Q2)
| File | Description |
| :--- | :--- |
| `value_iteration.py` | Implements the Bellman Optimality update loop. |
| `policy_iteration.py` | Implements the Evaluation-Improvement loop. |
| `induced_policy.py` | Extracts a deterministic greedy policy from Q-values. |
| **`Q2de.ipynb`** | **Orchestrator.** Runs the comparison and generates plots. |

## 🛠 Prerequisites

Ensure your environment has the following dependencies:
```bash
pip install numpy matplotlib
```

## 🚀 How to Run

1. **Verify Files:** Ensure all `.py` files are in the parent folder of the `Q2` folder, where `Q2de.ipynb` is located.
2. **Launch Notebook:** Open `Q2de.ipynb` in VS Code or Jupyter.
3. **Execution Steps:**
    - **Cell 1:** Run the imports. It is recommended to keep `%autoreload 2` enabled to catch changes in your `.py` modules.
    - **Cell 2:** Run the comparison logic. This will iterate through $N \in \{4, 6, 8, 10, 12\}$, executing both VI and PI for each.
    - **Cell 3:** Execute the plotting code to generate the $L_1$ difference and runtime comparison.

## 📈 Expected Results

- **$L_1$ Difference:** You should observe that the value function difference between VI and PI is near zero (within floating-point precision/epsilon), confirming they converge to the same $V^*$.
- **Runtime Comparison:** While Policy Iteration typically takes fewer "outer" iterations, Value Iteration may be faster in "wall-clock time" per step as it avoids the nested iterative evaluation required by PI.