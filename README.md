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
