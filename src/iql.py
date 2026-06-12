import random
import numpy as np
from algorithms import MARLAlgorithm


class IQL(MARLAlgorithm):

    def __init__(self, agent_id, game, gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id
        self.game = game
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table = np.zeros((game.num_states, game.num_actions))  # S x A (acciones individuales)
        self.metrics = {"td_error": []}

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def learn(self, joint_action, rewards, state, next_state):
        # IQL solo usa su accion y su recompensa
        action = joint_action[self.agent_id]
        reward = rewards[self.agent_id]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.metrics["td_error"].append(td_error)

    def select_action(self, state, train=True):
        if train and self.rng.random() < self.epsilon:
            return self.rng.choice(range(self.game.num_actions))
        else:
            return int(np.argmax(self.q_table[state]))

    def explain(self, state=0):
        return {"algorithm": "IQL", "agent_id": self.agent_id,
                "q_values": self.q_table[state].tolist()}
