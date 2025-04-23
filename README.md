# 🚀 Algorithmic Trading using DDDQN (Dueling Double Deep Q-Network)

<div align="center">

![Python](https://img.shields.io/badge/python-v3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Bitcoin](https://img.shields.io/badge/Bitcoin-Trading-gold.svg)

*AI-powered Bitcoin trading using advanced Deep Reinforcement Learning*
</div>

---

##  Overview

This project implements a **Dueling Double Deep Q-Network (DDDQN)** for algorithmic Bitcoin trading. The AI agent learns optimal trading strategies by analyzing historical Bitcoin price data from 2020-2025, making intelligent buy/sell/hold decisions to maximize profits while managing risk.

### Why DDDQN?

- **Dueling Architecture**: Separates state value and action advantage estimation
- **Double Q-Learning**: Reduces overestimation bias in Q-value updates  
- **Experience Replay**: Learns from past experiences more efficiently
- **Target Networks**: Provides stable learning targets

## ✨ Features

###  **Intelligent Trading Agent**
- Advanced DDDQN architecture for optimal decision making
- Dynamic position sizing and risk management
- Real-time Bitcoin price action analysis
- Adaptive learning from market patterns

### **Market Analysis**
- Comprehensive Bitcoin price data (2020-2025)
- Technical indicator integration
- Price pattern recognition
- Volatility-based position sizing

###  **Risk Management**
- Maximum position limits
- Stop-loss mechanisms
- Drawdown protection
- Portfolio value tracking

##  Quick Start

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

### Project Structure

```
 Algorithmic-Trading-using-DeepRL/
├── DDDQN/
│   ├── agent.py              # DDDQN agent implementation
│   ├── environment.py        # Trading environment
│   ├── main.py              # Main training/testing script
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── notebook.ipynb       # Analysis and visualization
├── bitcoin_2020_2025.csv   # Historical Bitcoin data
└── requirements.txt         # Python dependencies
```

### Running the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/AryanK05/Algorithmic-Trading-using-DeepRL.git
   cd Algorithmic-Trading-using-DeepRL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the DDDQN agent**
   ```bash
   cd DDDQN
   python main.py
   ```

4. **Explore the analysis**
   ```bash
   jupyter notebook notebook.ipynb
   ```

##  Architecture

The core of this project is the **Dueling Double Deep Q-Network (DDDQN)**. This architecture enhances the standard Deep Q-Network (DQN) by:

1.  **Dueling Network Structure**:
    * It features two separate streams within the neural network: one for estimating the **state value function ($V(s)$)** and another for estimating the **action advantage function ($A(s, a)$)**.
    * The state value ($V(s)$) represents how good it is to be in a particular state.
    * The advantage ($A(s,a)$) represents how much better taking a specific action ($a$) is compared to other actions in state ($s$).
    * These are combined to get the final Q-values: $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A(s, \cdot)))$.
    * This allows the model to learn which states are valuable without having to learn the effect of each action for each state, leading to better policy evaluation in states where actions have similar values.

2.  **Double Q-Learning**:
    * Addresses the overestimation bias of Q-values in traditional DQN.
    * It uses two networks: an **online network** (for selecting the best action in the next state) and a **target network** (for evaluating the Q-value of that action).
    * The target for the Q-value update is calculated as: $y_t = r_t + \gamma \cdot Q_{\text{target}}(s_{t+1}, \arg\max_{a'} Q_{\text{online}}(s_{t+1}, a'))$.
    * This decoupling of action selection from value estimation helps in achieving more accurate Q-value estimates.

3.  **Experience Replay**:
    * Stores transitions (state, action, reward, next state, done) in a replay buffer.
    * During training, mini-batches are randomly sampled from this buffer.
    * This breaks the correlation between consecutive samples, leading to more stable and efficient learning, and allows for reuse of past experiences.

4.  **Target Networks**:
    * A separate target network (a periodically updated copy of the online network) is used to generate the target Q-values for the Bellman equation.



###  Trading Performance

The DDDQN agent demonstrates strong performance on Bitcoin trading:

| Metric | Performance |
|--------|-------------|
| **Dataset Period** | 2020 - 2025 |
| **Action Space** | Buy (0), Sell (1), Hold (2) |
| **State Features** | OHLCV + Technical Indicators |
| **Convergence** | ~500 episodes |

###  Key Advantages

- **Reduced Overestimation**: Double Q-learning prevents Q-value inflation
- **Better Value Estimation**: Dueling architecture improves state value assessment  
- **Stable Learning**: Target networks provide consistent learning targets
- **Sample Efficiency**: Experience replay maximizes data utilization

## Data

The project uses historical Bitcoin data (`bitcoin_2020_2025.csv`) containing:
- **Timestamp**: Date and time information
- **OHLCV**: Open, High, Low, Close, Volume data
- **Period**: 2020-2025 market data for training and testing

</div>