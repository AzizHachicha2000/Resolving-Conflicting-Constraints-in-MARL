# 🛡️ Layered Safe Multi-Agent Reinforcement Learning (MARL)

## 📌 Overview
This project implements a **Layered Safe MARL framework** for safe multi-robot navigation.  
The framework integrates **multi-agent reinforcement learning (MARL)** with **safety filters** to prevent collisions while maintaining efficiency.  

Traditional MARL lacks safety guarantees, while classical control methods (e.g., Control Barrier Functions, Reachability Analysis) struggle with scalability.  
This project bridges both by introducing **layered safety mechanisms** that resolve conflicts between agents in real-time.  

The system is validated on:
- **Crazyflie drone hardware experiments**  
- **High-density Advanced Aerial Mobility (AAM) simulations**  

---

## ⚙️ Model Workflow
The framework is composed of **three layers**:

1. **Learning Layer (MARL)**  
   - Agents learn navigation policies to minimize multi-agent conflicts.  
   - Uses a temporal transformer architecture with GNN-based communication.  

2. **Prioritization Layer**  
   - Detects agents entering **engagement distance**.  
   - Prioritizes the most urgent conflicts based on risk and safety margin.  

3. **Safety Filtering Layer**  
   - Uses **Control Barrier-Value Functions (CBVF)** to filter unsafe actions.  
   - Applies `clip()` operations to keep actions within safety bounds.  

---

## 📐 Mathematical Formulation

### Action Filtering
The corrected action `u_filtered` is obtained via:
\[
u_{\text{filtered}} = \text{clip}(u_{\text{MARL}}, u_{\text{min}}, u_{\text{max}})
\]

Where:
\[
\text{clip}(x,a,b)= 
\begin{cases}
a & \text{if } x < a \\
x & \text{if } a \leq x \leq b \\
b & \text{if } x > b
\end{cases}
\]

### Safety Filter Weighting
The blending factor λ is:
\[
\lambda =
\begin{cases}
1 & \text{if safety violation detected (override)} \\
0 & \text{if action is safe (no filter applied)} \\
\in (0,1) & \text{if partial correction needed}
\end{cases}
\]

Final action:
\[
u_{\text{final}} = (1-\lambda) \cdot u_{\text{MARL}} + \lambda \cdot u_{\text{safe}}
\]

---

## 🧠 Training Process

1. **Environment Setup**
   - Multi-agent navigation tasks with obstacles.  
   - Agents must reach waypoints while avoiding collisions.  

2. **Policy Learning (MARL)**
   - Each agent learns a policy π via reinforcement learning:  
   \[
   \pi(a|s;\theta) = P(a|s;\theta)
   \]  
   where `s` = state, `a` = action, `θ` = parameters.  

3. **Reward Function**
   \[
   R = R_{\text{progress}} - \alpha R_{\text{collision}} - \beta R_{\text{deviation}}
   \]  

   - Encourages reaching goals quickly.  
   - Penalizes collisions and unnecessary detours.  

4. **Optimization**
   - Policy updated via gradient ascent:  
   \[
   \theta \leftarrow \theta + \eta \nabla_\theta J(\theta)
   \]  

---

## 🧩 Code Snippets

### Clipping Action
```python
def clip(x, a, b):
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x
Safety Filtering
python
Copier
Modifier
def safety_filter(u_marl, u_safe, violation):
    if violation:
        λ = 1  # full override
    else:
        λ = 0  # no override
    return (1 - λ) * u_marl + λ * u_safe
Reward Function
python
Copier
Modifier
def reward(progress, collision, deviation, alpha=1.0, beta=0.5):
    return progress - alpha * collision - beta * deviation
🔄 Working Flow
mermaid
Copier
Modifier
flowchart TD
    A[Environment Input] --> B[MARL Policy π]
    B --> C[Predicted Action u_MARL]
    C --> D[Conflict Detection]
    D -->|Safe| E[Execute Action]
    D -->|Unsafe| F[Safety Filter CBVF]
    F --> G[Corrected Action u_safe]
    G --> E
    E --> H[Agent State Update]
    H --> A
🚀 Results
Hardware: Successfully tested on Crazyflie drones.

Simulations: Scalable to high-density aerial mobility scenarios.

Performance:

Significant conflict reduction.

Maintains near-optimal efficiency (travel time & distance).

📂 Repository Structure
bash
Copier
Modifier
├── models/             # Trained MARL models
├── scripts/            # Training and evaluation scripts
├── utils/              # Helper functions (math, plotting, logging)
├── results/            # Logs, plots, evaluation results
├── README.md           # Project documentation
└── report.pdf          # Detailed mathematical and technical report
