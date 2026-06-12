import os
import time
from tqdm import tqdm
from algorithms import JALGT
from iql import IQL
from solution_concepts import ParetoSolutionConcept
from game_model import GameModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
# from utils import draw_history


def obs_to_state(obs):
    # Esta representación base asume observaciones de radio 1 (matrices 3x3).
    # Está pensada para el alcance obligatorio de la práctica; podéis cambiarla
    # si queréis experimentar con estados más complejos.
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]

    # Representación del objetivo:
    #  Ocupa 2 bits
    #  0 si el objetivo está arriba, diagonal arriba-izquierda o diagonal arriba-derecha
    #  1 si el objetivo está abajo, diagonal abajo-izquierda o diagonal abajo-derecha
    #  2 si el objetivo está a la izquierda (no en diagonal)
    #  3 si el objetivo está a la derecha (no en diagonal)
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3

    # Representación de los obstáculos:
    #  Shift de 2^6, ocupando 4 bits
    #  2^9 si hay un obstáculo arriba (no diagonal)
    #  2^8 si hay un obstáculo a la izquierda (no diagonal)
    #  2^7 si hay un obstáculo a la derecha (no diagonal)
    #  2^6 si hay un obstáculo abajo (no diagonal)
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6

    # Representación de los otros agentes:
    #  Shift de 2^2, ocupando 4 bits
    #  2^5 si hay un agente arriba (no diagonal)
    #  2^4 si hay un agente a la izquierda (no diagonal)
    #  2^3 si hay un agente a la derecha (no diagonal)
    #  2^2 si hay un agente abajo (no diagonal)
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2

    return int(obstacles + agents + target)


class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, joint_action):
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:  # Penalización por tardar más en llegar
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos


def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=seed,
                             max_episode_steps=config["episode_length"],
                             obs_radius=1,
                             on_target="finish",
                             render_mode=None)
    animation_config = AnimationConfig(directory=config["renders"],  # Dónde se guardarán las imágenes
                                       static=False,
                                       show_agents=True,
                                       egocentric_idx=None,  # Punto de vista
                                       save_every_idx_episode=config["save_every"],  # Guardar cada save_every episodios
                                       show_border=True,
                                       show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)  # Añadimos nuestra función de recompensa


def compute_epsilon(config, global_episode):
    # Usamos un decay global por episodio para que el estudio de epsilon sea
    # fácil de interpretar y comparable entre algoritmos.
    total_episodes = config["epochs"] * config["episodes_per_epoch"]
    progress = global_episode / max(total_episodes - 1, 1)
    return config["epsilon_max"] - progress * (config["epsilon_max"] - config["epsilon_min"])


def train_episode(env, algorithms, game, epsilon):
    for algorithm in algorithms:
        algorithm.set_epsilon(epsilon)

    observations, infos = env.reset()
    terminated = [False] * game.num_agents
    truncated = [False] * game.num_agents
    total_rewards = [0] * game.num_agents
    td_errors = []
    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    while not all(terminated) and not all(truncated):
        actions = tuple(algorithms[i].select_action(states[i]) for i in range(game.num_agents))
        observations, rewards, terminated, truncated, infos = env.step(actions)
        next_states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
        for i in range(game.num_agents):
            algorithms[i].learn(actions, rewards, states[i], next_states[i])
            td_errors.append(algorithms[i].metrics["td_error"][-1])
        total_rewards = [total_rewards[i] + rewards[i] for i in range(game.num_agents)]
        states = next_states

    return total_rewards, td_errors


def evaluate_episode(env, algorithms, game):
    observations, infos = env.reset()
    terminated = [False] * game.num_agents
    truncated = [False] * game.num_agents
    total_rewards = [0] * game.num_agents
    states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    while not all(terminated) and not all(truncated):
        actions = tuple(algorithms[i].select_action(states[i], train=False) for i in range(game.num_agents))
        observations, rewards, terminated, truncated, infos = env.step(actions)
        total_rewards = [total_rewards[i] + rewards[i] for i in range(game.num_agents)]
        states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

    return total_rewards


# construir agentes JAL-GT o IQL con semilla determinista (run_seed*10 + agent_id)
def build_agents(algorithm_cls, config, game, run_seed):
    agents = []
    for agent_id in range(game.num_agents):
        seed = run_seed * 10 + agent_id
        if algorithm_cls is JALGT:
            agents.append(JALGT(agent_id, game, config["solution_concept"](), gamma=config["gamma"],
                                alpha=config["learning_rate"], epsilon=config["epsilon_max"], seed=seed))
        else:
            agents.append(IQL(agent_id, game, gamma=config["gamma"], alpha=config["learning_rate"],
                              epsilon=config["epsilon_max"], seed=seed))
    return agents


if __name__ == '__main__':
    exp_config = {
        "num_agents": 2,  # Número de agentes
        "size": 4,  # Tamaño del mapa (valor de anchura y valor de altura)
        "maps": 10,  # Número de mapas a entrenar y evaluar (se repiten si episodios > mapas)
        "num_states": 16 * 16 * 4,  # Obstacle representation x Agent representation x Target representation
        "epochs": 120,  # Cada epoch es un entrenamiento de un número de episodios y una evaluación
        "episodes_per_epoch": 10,  # Número mínimo de episodios por epoch de entrenamiento
        "episode_length": 16,  # Número máximo de pasos por episodio, se trunca si se excede
        "obstacle_density": 0.3,  # Probabilidad de tener un obstáculo en el mapa
        "save_every": None,  # Frecuencia con que se guarda el SVG con la animación de la ejecución
        "learning_rate": 0.1,  # alpha
        "gamma": 0.95,  # factor de descuento (uno de los hiperparámetros a estudiar)
        "epsilon_max": 1,  # epsilon inicial del entrenamiento
        "epsilon_min": 0.1,  # cota mínima de epsilon
        "renders": "results1/renders/",  # directorio donde generar las animaciones
        "solution_concept": ParetoSolutionConcept, 
        "seeds": [0, 1, 2],  # semillas a promediar para tener media y desviación
        "tag": "env_4x4_med",  # nombre para los ficheros de salida (CSV, figuras, SVG)
    }

    # Creamos el directorio de renders (y el de figuras) si no existen ya.
    os.makedirs("results1/figures", exist_ok=True)
    os.makedirs(exp_config["renders"], exist_ok=True)

    # guardamos aquí las curvas de cada algoritmo
    results = {}

    # comparamos JAL-GT contra IQL con la misma configuración
    for algorithm_name, algorithm_cls in (("JAL-GT", JALGT), ("IQL", IQL)):

        # repetimos el entrenamiento para cada semilla, para tener media y desviación
        reward_per_seed, td_error_per_seed, success_per_seed = [], [], []
        t0 = time.time()
        for run_seed in exp_config["seeds"]:

            # Modelo de juego y algoritmos (uno para cada agente)
            game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"],
                             num_actions=5)  # STAY, UP, DOWN, LEFT, RIGHT
            algorithms = build_agents(algorithm_cls, exp_config, game, run_seed)

            # Variables para almacenar métricas
            reward_per_epoch = []
            td_error_per_epoch = []
            success_per_epoch = []
            td_seen = 0  # para quedarnos sólo con los TD errors nuevos de este epoch

            pbar = tqdm(range(exp_config["epochs"]))  # Barra de progreso
            for epoch in pbar:
                pbar.set_description(f"{algorithm_name} seed={run_seed}")
                all_eval_rewards = []
                all_success = []

                # Entrenamiento
                ###############
                for ep in range(exp_config["episodes_per_epoch"]):
                    pbar.set_postfix({'modo': 'entrenamiento', 'episodio': ep})
                    global_episode = epoch * exp_config["episodes_per_epoch"] + ep
                    epsilon = compute_epsilon(exp_config, global_episode)
                    env = create_env(config=exp_config, seed=ep % exp_config["maps"])
                    train_episode(env, algorithms, game, epsilon)
                # TD medio de los updates de este epoch
                new_td_errors = algorithms[0].metrics["td_error"][td_seen:]
                td_seen += len(new_td_errors)
                td_error_per_epoch.append(float(np.mean(np.abs(new_td_errors))) if new_td_errors else 0.0)

                # Evaluación
                ############
                evaluation_episodes = exp_config["maps"]
                for ep in range(evaluation_episodes):
                    pbar.set_postfix({'modo': 'evaluación...', 'episodio': ep})
                    env = create_env(config=exp_config, seed=ep)  # Reaprovechamos mapas del entrenamiento
                    total_rewards = evaluate_episode(env, algorithms, game)
                    all_eval_rewards.append(sum(total_rewards))
                    # éxito = fracción de agentes que han llegado a su objetivo (recompensa positiva)
                    all_success.append(sum(1 for r in total_rewards if r > 0) / exp_config["num_agents"])
                reward_per_epoch.append(np.mean(all_eval_rewards))
                success_per_epoch.append(np.mean(all_success))

            reward_per_seed.append(reward_per_epoch)
            td_error_per_seed.append(td_error_per_epoch)
            success_per_seed.append(success_per_epoch)

            # Guardamos animaciones: sólo para la primera semilla y una vez al final
            # (para no generar miles de SVGs)
            if run_seed == exp_config["seeds"][0]:
                env = create_env(config=exp_config, seed=0)
                evaluate_episode(env, algorithms, game)
                for agent_i in range(exp_config["num_agents"]):
                    alg_tag = algorithm_name.replace("-", "")
                    env.save_animation(f"{exp_config['renders']}/{exp_config['tag']}_{alg_tag}-agent{agent_i}.svg",
                                       AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True))

        results[algorithm_name] = {"reward": np.array(reward_per_seed), "success": np.array(success_per_seed),
                                    "td": np.array(td_error_per_seed), "time": time.time() - t0}

    # figuras: una por métrica
    for metric, ylabel in (("reward", "Recompensa colectiva"), ("success", "Tasa de éxito"),
                           ("td", "|TD error| medio")):
        plt.figure(figsize=(9, 5))
        for algorithm_name, r in results.items():
            mean, std = r[metric].mean(0), r[metric].std(0)
            x = np.arange(1, len(mean) + 1)
            plt.plot(x, mean, label=algorithm_name)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.title(f"{exp_config['tag']} · {ylabel}")
        plt.xlabel("Epoch"); plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"results/figures/{exp_config['tag']}_{metric}.png", dpi=130)
        plt.close()

    # CSV resumen: media del último 10% de epochs
    tail = max(1, exp_config["epochs"] // 10)
    pd.DataFrame([{"algoritmo": algorithm_name,
                   "recompensa_final": float(r["reward"].mean(0)[-tail:].mean()),
                   "exito_final": float(r["success"].mean(0)[-tail:].mean()),
                   "tiempo_s": round(r["time"], 1)}
                  for algorithm_name, r in results.items()]).to_csv(f"results/{exp_config['tag']}_summary.csv", index=False)