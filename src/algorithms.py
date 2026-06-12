import random

import numpy as np

from solution_concepts import SolutionConcept
from game_model import GameModel

from tinynn.core.layer import Dense, ReLU
from tinynn.core.loss import MSE, SoftmaxCrossEntropy
from tinynn.core.initializer import Zeros
from tinynn.core.model import Model
from tinynn.core.net import Net
from tinynn.core.optimizer import SGD
import abc


class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, joint_action, rewards, state: int, next_state: int):
        pass

    @abc.abstractmethod
    def explain(self, state=0):
        pass

    @abc.abstractmethod
    def select_action(self, state, train=True):
        pass


class JALGT(MARLAlgorithm):
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        # Q: N x S x AS
        self.q_table = np.zeros((self.game.num_agents, self.game.num_states,
                                 len(self.game.action_space)))
        print("Initialized Q-table with shape:", self.q_table.shape)
        # Política conjunta por defecto: distribución uniforme respecto
        # de las acciones conjuntas, para cada acción (pi(a | s))
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

    def value(self, agent_id, state):
        value = 0
        for idx, joint_action in enumerate(self.game.action_space):
            payoff = self.q_table[agent_id][state][idx]
            joint_probability = np.prod([self.joint_policy[i][state][joint_action[i]]
                                         for i in range(self.game.num_agents)])
            value += payoff * joint_probability
        return value

    def update_policy(self, agent_id, state):
        self.joint_policy[agent_id][state] = self.solution_concept.solution_policy(agent_id, state, self.game,
                                                                                   self.q_table)

    def learn(self, joint_action, rewards, state, next_state):
        joint_action_index = self.game.action_space_index[joint_action]
        for agent_id in range(self.game.num_agents):
            agent_reward = rewards[agent_id]
            agent_game_value_next_state = self.value(agent_id, next_state)
            agent_q_value = self.q_table[agent_id][state][joint_action_index]
            td_target = agent_reward + self.gamma * agent_game_value_next_state - agent_q_value
            self.q_table[agent_id][state][joint_action_index] += self.alpha * td_target
            self.update_policy(agent_id, state)
            # Guardamos el error de diferencia temporal para estadísticas posteriores
            self.metrics['td_error'].append(td_target)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def solve(self, agent_id, state):
        return self.joint_policy[agent_id][state]

    def select_action(self, state, train=True):
        if train:
            if self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
            else:
                probs = self.solve(self.agent_id, state)
                np.random.seed(self.rng.randint(0, 10000))
                return np.random.choice(range(self.game.num_actions), p=probs)
        else:
            return np.argmax(self.solve(self.agent_id, state))

    def explain(self, state=0):
        return self.solution_concept.debug(self.agent_id, state, self.game, self.q_table)


def one_hot(state, max_states):
    one_hot_vector = np.zeros(max_states)
    one_hot_vector[state] = 1
    return np.array([one_hot_vector], dtype=float)

def normal_form_obs_to_state(observation):
    # Sólo un estado (juego en forma normal)
    return 0

def train(env, game, f_obs_to_state, algorithms):
    cumulative_rewards = [[0, 0]]
    actions_played = [[], []]
    all_agents = range(game.num_agents)

    observations, _ = env.reset()
    states = [f_obs_to_state(observations[f"player_{i}"]) for i in all_agents]

    while env.agents:
        joint_action = tuple([algorithms[i].select_action(states[i]) for i in all_agents])
        [actions_played[i].append(joint_action[i]) for i in all_agents]
        pettingzoo_joint_action = {f"player_{i}": joint_action[i] for i in all_agents}

        observations, rewards, terminations, truncations, infos = env.step(pettingzoo_joint_action)

        observations = [observations[f"player_{i}"] for i in all_agents]
        rewards = [rewards[f"player_{i}"] for i in all_agents]
        new_states = [f_obs_to_state(observations[i]) for i in all_agents]
        [algorithms[i].learn(joint_action, rewards, states[i], new_states[i])
         for i in all_agents]
        cumulative_rewards.append([cumulative_rewards[-1][i] + rewards[i] for i in all_agents])
        states = new_states

    return cumulative_rewards, actions_played

def normalize(x):
    x = np.array(x, dtype=float, copy=True)
    min_val = np.min(x)
    if min_val < 0:
        x -= min_val
    total_sum = np.sum(x)
    if total_sum == 0:
        return np.ones(len(x), dtype=float) / len(x)
    return x / total_sum


class WoLFPHC(MARLAlgorithm):
    # 
#   def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept, gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
    def __init__(self, agent_id, game: GameModel, learning_rate_loss, learning_rate_win,
                 hidden_sizes=(64, 32), gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        assert(learning_rate_loss > learning_rate_win)
        self.agent_id = agent_id
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate_win = learning_rate_win
        self.learning_rate_loss = learning_rate_loss
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # Build Q-network with configurable hidden sizes
        net_q_layers = []
        for h in hidden_sizes:
            net_q_layers.append(Dense(h))
            net_q_layers.append(ReLU())
        net_q_layers.append(Dense(game.num_actions))
        net_q = Net(net_q_layers)
        self.q_model = Model(net=net_q, loss=MSE(), optimizer=SGD(lr=alpha))
        self.visit_counts = np.zeros(game.num_states)
        # Policy network
        net_pi_layers = []
        for h in hidden_sizes:
            net_pi_layers.append(Dense(h))
            net_pi_layers.append(ReLU())
        net_pi_layers.append(Dense(game.num_actions))
        net_pi = Net(net_pi_layers)
        self.policy = Model(net=net_pi, loss=SoftmaxCrossEntropy(), optimizer=SGD(lr=0.1))
        # Average policy network
        net_avg_pi_layers = []
        for h in hidden_sizes:
            net_avg_pi_layers.append(Dense(h))
            net_avg_pi_layers.append(ReLU())
        net_avg_pi_layers.append(Dense(game.num_actions))
        net_avg_pi = Net(net_avg_pi_layers)
        self.average_policy = Model(net=net_avg_pi, loss=SoftmaxCrossEntropy(), optimizer=SGD(lr=0.1))
        self.metrics = {'td_error': [], 'loss': []}



    def policy_probs(self, scalar_state):
        logits = self.policy.forward(one_hot(scalar_state, self.game.num_states))[0]
        return normalize(np.exp(logits - np.max(logits)))



    def average_policy_probs(self, scalar_state):
        logits = self.average_policy.forward(one_hot(scalar_state, self.game.num_states))[0]
        return normalize(np.exp(logits - np.max(logits)))
    

    
    def update_q(self, q_model, state, action, reward, next_value):
        prediction = q_model.forward(state)[0]
        agent_q_value = prediction[action]
        td_target = reward + self.gamma * next_value
        td_error = td_target - agent_q_value
        target = prediction.copy()
        target[action] = td_target
        loss, grads = q_model.backward(np.array([prediction]), np.array([target]))
        q_model.apply_grads(grads)
        return td_error, loss
    


    def update_average_policy(self, scalar_state):
        state = one_hot(scalar_state, self.game.num_states)
        avg_pi_logits = self.average_policy.forward(state)
        avg_pi_s = self.average_policy_probs(scalar_state)
        current_pi_s = self.policy_probs(scalar_state)
        target_avg_pi_s = avg_pi_s.copy()
        for a_i in range(self.game.num_actions):
            pi_a_i = current_pi_s[a_i]
            # Sumamos el gradiente del error: diferencia entre política actual y política media,
            # ponderado por el número de visitas en el estado para converger en un valor estable
            target_avg_pi_s[a_i] += (pi_a_i - avg_pi_s[a_i]) / self.visit_counts[scalar_state]
        target_avg_pi_s = normalize(target_avg_pi_s)
        # Actualizamos la red neuronal sobre logits con cross-entropy
        _, grads = self.average_policy.backward(avg_pi_logits, np.array([target_avg_pi_s]))
        self.average_policy.apply_grads(grads)



    def update_policy(self, scalar_state):
        state = one_hot(scalar_state, self.game.num_states)

        current_policy_logits = self.policy.forward(state)
        current_policy_probs = self.policy_probs(scalar_state)
        current_avg_pi_probs = self.average_policy_probs(scalar_state)
        q_values_s = self.q_model.forward(state)[0]

        # Calculamos delta (ecuación 6.46)
        policy_value = np.sum(np.multiply(current_policy_probs, q_values_s))
        avg_pi_value = np.sum(np.multiply(current_avg_pi_probs, q_values_s))
        if policy_value > avg_pi_value:
            delta = self.learning_rate_win
        else:
            delta = self.learning_rate_loss

        # Calculamos delta para (s, a_i) (ecuación 6.45)
        delta_all = [min(current_policy_probs[a_i], delta / (self.game.num_actions - 1)) for a_i in range(self.game.num_actions)]

        # Comprobamos qué acción es la mejor según Q y con qué valor
        max_q = np.max(q_values_s)
        best_actions = [a_i for a_i in range(self.game.num_actions) if np.isclose(q_values_s[a_i], max_q)]
        target_policy_s = current_policy_probs.copy()
        total_negative_mass = sum(delta_all[a_i] for a_i in range(self.game.num_actions) if a_i not in best_actions)
        positive_update = total_negative_mass / len(best_actions)

        # Recorremos las acciones, aplicando delta
        for a_i in range(self.game.num_actions):
            if a_i in best_actions:
                # Mejores acciones, aplicamos positivamente
                target_policy_s[a_i] += positive_update
            else:
                # Demás acciones, aplicamos negativamente
                target_policy_s[a_i] -= delta_all[a_i]
        # Actualizamos la red neuronal sobre logits con cross-entropy
        target_policy_s = normalize(target_policy_s)
        loss, grads = self.policy.backward(current_policy_logits, np.array([target_policy_s]))
        self.policy.apply_grads(grads)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def learn(self, joint_action, rewards, state, next_state):
        # Sólo consideramos lo que respecta a nuestro agente
        agent_action = joint_action[self.agent_id]
        agent_reward = rewards[self.agent_id]
        one_hot_state = one_hot(state, self.game.num_states)
        one_hot_next_state = one_hot(next_state, self.game.num_states)
        next_value = np.max(self.q_model.forward(one_hot_next_state)[0])

        # Actualizamos valor Q
        td_error, loss = self.update_q(self.q_model, one_hot_state, agent_action, agent_reward, next_value)

        # Actualizamos política media
        self.visit_counts[state] += 1
        self.update_average_policy(state)
        self.update_policy(state)

        # Guardamos el error de diferencia temporal y la pérdida de la red neuronal para estadísticas posteriores
        self.metrics['td_error'].append(abs(td_error))
        self.metrics['loss'].append(abs(loss))



    def select_action(self, state, train=True):
        if train and self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
        else:
            probs = self.policy_probs(state)
            return self.np_rng.choice(range(self.game.num_actions), p=probs)

    def explain(self, state=0):
        """Return a small diagnostic dict with policy, average policy and Q-values for `state`."""
        try:
            q_vals = self.q_model.forward(one_hot(state, self.game.num_states))[0]
        except Exception:
            q_vals = None
        return {
            "agent_id": self.agent_id,
            "state": state,
            "policy": self.policy_probs(state).tolist(),
            "average_policy": self.average_policy_probs(state).tolist(),
            "q_values": None if q_vals is None else q_vals.tolist(),
        }