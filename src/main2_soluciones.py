import os
from tqdm import tqdm
from algorithms import JALGT
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history

import time

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


def build_algorithms(config, game):
    algorithm_cls = config["algorithm_cls"]
    solution_concept_cls = config.get("solution_concept")
    algorithm_kwargs = dict(config.get("algorithm_kwargs", {}))

    algorithms = []
    for agent_id in range(game.num_agents):
        kwargs = dict(algorithm_kwargs)
        kwargs.setdefault("epsilon", config["epsilon_max"])
        kwargs.setdefault("alpha", config["learning_rate"])
        kwargs.setdefault("seed", agent_id)
        if solution_concept_cls is not None:
            kwargs.setdefault("solution_concept", solution_concept_cls())
        algorithms.append(algorithm_cls(agent_id, game, **kwargs))
    return algorithms


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


if __name__ == '__main__':
    exp_config = {
        "num_agents": 2,  # Número de agentes
        "size": 4,  # Tamaño del mapa (valor de anchura y valor de altura)
        "maps": 10,  # Número de mapas a entrenar y evaluar (se repiten si episodios > mapas)
        "num_states": 16 * 16 * 4,  # Obstacle representation x Agent representation x Target representation
        "epochs": 200,  # Cada epoch es un entrenamiento de un número de episodios y una evaluación
        "episodes_per_epoch": 10,  # Número mínimo de episodios por epoch de entrenamiento
        "episode_length": 16,  # Número máximo de pasos por episodio, se trunca si se excede
        "obstacle_density": 0.1,  # Probabilidad de tener un obstáculo en el mapa
        "save_every": None,  # Frecuencia con que se guarda el SVG con la animación de la ejecución
        "learning_rate": 0.01,  # alpha
        "epsilon_max": 1,  # epsilon inicial del entrenamiento
        "epsilon_min": 0.1,  # cota mínima de epsilon
        "renders": "renders/",  # directorio donde generar las animaciones
        "algorithm_cls": JALGT,  # Aquí podéis conectar IQL u otro algoritmo.
        "algorithm_kwargs": {},
        "solution_concept": MinimaxSolutionConcept
    }

    #ParetoSolutionConcept
    #NashSolutionConcept
    #MinimaxSolutionConcept

    # Creamos el directorio de renders si no existe ya.
    os.makedirs(exp_config["renders"], exist_ok=True)

    # Modelo de juego y algoritmos (uno para cada agente)
    game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"],
                     num_actions=5)  # STAY, UP, DOWN, LEFT, RIGHT
    algorithms = build_algorithms(exp_config, game)

    # Variables para almacenar métricas
    reward_per_epoch = []
    td_error_per_epoch = []

    #rewards_ag = [[] for _ in range(exp_config["num_agents"])]
    total_complet = [0]*exp_config["num_agents"]
    total_aproved = [0]*exp_config["num_agents"]
    totals_by_map = [[0 for _ in range(exp_config["maps"])]for _ in range(exp_config["num_agents"])]
    total_time = 0
    total_reward_by_agent = [0]*exp_config["num_agents"]

    
    totals_by_map_final = [[0 for _ in range(exp_config["maps"])]for _ in range(exp_config["num_agents"])]
    total_final = [0]*exp_config["num_agents"]

    pbar = tqdm(range(exp_config["epochs"]))  # Barra de progreso
    for epoch in pbar:
        all_eval_rewards = []
        all_td_errors = []
        
        total_final_aux = [0]*exp_config["num_agents"]

        # Entrenamiento
        ###############
        iter_start = time.perf_counter()
        for ep in range(exp_config["episodes_per_epoch"]):
            pbar.set_postfix({'modo': 'entrenamiento', 'episodio': ep})
            global_episode = epoch * exp_config["episodes_per_epoch"] + ep
            epsilon = compute_epsilon(exp_config, global_episode)
            env = create_env(config=exp_config, seed=ep % exp_config["maps"])
            _, td_errors = train_episode(env, algorithms, game, epsilon)
            all_td_errors.extend(td_errors)
        td_error_per_epoch.append(sum(all_td_errors))
        total_time += time.perf_counter() - iter_start

        # Evaluación
        ############
        evaluation_episodes = exp_config["maps"]
        all_eval_rewards = []
        for ep in range(evaluation_episodes):
            pbar.set_postfix({'modo': 'evaluación...', 'episodio': ep})
            env = create_env(config=exp_config, seed=ep)  # Reaprovechamos mapas del entrenamiento
            total_rewards = evaluate_episode(env, algorithms, game)
            # Guardamos animaciones
            for agent_i in range(exp_config["num_agents"]):
                #rewards_ag[agent_i].append(total_rewards[agent_i])
                if (total_rewards[agent_i] == 1):
                    total_complet[agent_i] += 1
                    total_final_aux[agent_i] += 1
                if (total_rewards[agent_i] < 1 and total_rewards[agent_i] > 0):
                    total_aproved[agent_i] += 1
                    total_final_aux[agent_i] += 1
                totals_by_map[agent_i][ep] += total_rewards[agent_i]
                total_reward_by_agent[agent_i] += total_rewards[agent_i]
                totals_by_map_final[agent_i][ep] = total_rewards[agent_i]

                solution_concept_name = exp_config["solution_concept"].__name__ if exp_config["solution_concept"] else "NoConcept"
                env.save_animation(f"{exp_config['renders']}/{solution_concept_name}-map{ep}-agent{agent_i}-epoch{epoch}.svg",
                                   AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True))
            all_eval_rewards.append(sum(total_rewards))
        pbar.set_description(f"Recompensa colectiva del último epoch = {'{:>6.6}'.format(str(sum(all_eval_rewards)))}")
        reward_per_epoch.append(sum(all_eval_rewards))
        total_final = total_final_aux

    draw_history(reward_per_epoch, "Recompensa colectiva")
    draw_history(td_error_per_epoch, "TD Error")
    print("Tiempo total entrenamiento: ", total_time)
    for agent_i in range(exp_config["num_agents"]):
        print("AGENTE ",agent_i)
        #draw_history(rewards_ag[agent_i], "Recompensa total agente")
        #draw_history(totals_by_map[agent_i], "Recompensa por mapas: ")
        print("Recompensas totales: ", total_complet[agent_i])
        print("Recompensas semitotales: ", total_aproved[agent_i])
        aux = round(total_reward_by_agent[agent_i], 2)
        print("Recompensas agente ", agent_i, " : ", aux)
        totals_by_map_round = [round(x, 2) for x in totals_by_map[agent_i]]
        print("Recompensas por mapa: ", totals_by_map_round)
        totals_by_map_final_round = [round(x, 2) for x in totals_by_map_final[agent_i]]
        print("Recompensa por mapa final: ", totals_by_map_final_round)
        aux2 = round(sum(totals_by_map_final[agent_i]), 2)
        print("Recompensa en el ultimo epoch : ", aux2)
        print("Cantidad objetivos final: ", round(total_final_aux[agent_i]/exp_config["maps"], 2))
