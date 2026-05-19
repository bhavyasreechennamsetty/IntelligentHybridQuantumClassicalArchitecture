import numpy as np
import random

class QLearningAgent:
    """
    Reinforcement Learning Agent for Orchestration style decisions.
    State: (LatencyState, ComplexityState)
    Actions: 0 (Classical), 1 (Quantum)
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {} # Map state_tuple -> [q_classical, q_quantum]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def get_state_key(self, latency, complexity):
        # Discretize state space
        l_state = "L_HIGH" if latency > 0.05 else "L_LOW"
        c_state = "C_HIGH" if complexity > 0.25 else "C_LOW"
        return (l_state, c_state)
        
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
            
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])
            
    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
            
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)
