# Advanced Intelligent Routing Algorithm

## Overview
This project implements an **Advanced Intelligent Routing Algorithm** designed to ensure reliable and secure data transmission in dynamic networks. It outperforms traditional protocols (like Dijkstra) by incorporating **Trust Models**, **Reinforcement Learning (Q-Learning)**, and **Quality of Service (QoS)** awareness.

## Key Features
1.  **Intelligent Trust Routing**: Avoids unreliable nodes by maintaining a dynamic trust score.
2.  **Reinforcement Learning (Q-Learning)**: Model-free agents learn optimal paths through exploration/exploitation.
3.  **Quality of Service (QoS)**: Prioritizes critical "Voice" traffic over "Data" during congestion.
4.  **Adversarial Defense**: robust against Blackhole, Grayhole, and On-Off attacks.
5.  **Interactive Dashboard**: Real-time visualization of network topology and routing logic.

## Project Structure
*   `main.py`: Core simulation comparing Standard vs Intelligent vs RL routing.
*   `network_sim.py`: Discrete-event network simulation engine (SimPy).
*   `routing.py`: Implementation of Dijkstra, Intelligent, and RL algorithms.
*   `trust_model.py`: Logic for calculating and updating node trust.
*   `rl_agent.py`: Q-Learning agent implementation.
*   `security.py`: Adversarial attack patterns.
*   `dashboard.py`: Streamlit-based web interface.

## How to Run

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Comparative Analysis
Simulates all algorithms and generates performance charts.
```bash
python compare_algos.py
```
*Output: `comparison_chart.png`*

### 3. Run Interactive Dashboard
Launch the web UI to visualize routing in real-time.
```bash
streamlit run dashboard.py
```

### 4. Run Specific Simulations
*   **Adversarial Attacks**: `python attack_sim.py`
*   **QoS Test**: `python qos_sim.py`
*   **RL Training**: `python main.py`

## Results
*   **Packet Delivery Ratio (PDR)**: Intelligent Routing achieves >95% PDR in hostile environments where Standard Routing fails (<60%).
*   **Security**: Successfully detects and isolates malicious nodes within seconds.
*   **Latency**: Maintains low latency for Voice traffic even under heavy congestion.

## License
MIT License
