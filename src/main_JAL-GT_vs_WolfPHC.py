import os
from tqdm import tqdm
from algorithms import JALGT, WoLFPHC
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
import numpy as np
from gymnasium import Wrapper
import matplotlib.pyplot as plt
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history


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



def run_experiment(exp_config):
    
    # Creamos el directorio de renders si no existe ya.
    os.makedirs(exp_config["renders"], exist_ok=True)

    num_states = exp_config["num_states"]

    # Modelo de juego y algoritmos (uno para cada agente)
    game = GameModel(num_agents=exp_config["num_agents"], num_states=num_states,
                        num_actions=5)  # STAY, UP, DOWN, LEFT, RIGHT
    algorithms = build_algorithms(exp_config, game)

    # Variables para almacenar métricas
    reward_per_epoch = []
    td_error_per_epoch = []

    pbar = tqdm(range(exp_config["epochs"]))  # Barra de progreso
    for epoch in pbar:
        all_eval_rewards = []
        all_td_errors = []

        # Entrenamiento
        ###############
        for ep in range(exp_config["episodes_per_epoch"]):
            pbar.set_postfix({'modo': 'entrenamiento', 'episodio': ep})
            global_episode = epoch * exp_config["episodes_per_epoch"] + ep
            epsilon = compute_epsilon(exp_config, global_episode)
            env = create_env(config=exp_config, seed=ep % exp_config["maps"])
            _, td_errors = train_episode(env, algorithms, game, epsilon)
            all_td_errors.extend(td_errors)
        td_error_per_epoch.append(sum(all_td_errors))

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
                solution_concept_name = exp_config["solution_concept"].__name__ if exp_config["solution_concept"] else "NoConcept"
                env.save_animation(f"{exp_config['renders']}/{solution_concept_name}-map{ep}-agent{agent_i}-epoch{epoch}.svg",
                                    AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True))
            all_eval_rewards.append(sum(total_rewards))
        pbar.set_description(f"Recompensa colectiva del último epoch = {'{:>6.6}'.format(str(sum(all_eval_rewards)))}")
        reward_per_epoch.append(sum(all_eval_rewards))
    return reward_per_epoch, td_error_per_epoch


def draw_comparative_history(rewards_jalgt, rewards_wolf, td_jalgt, td_wolf, plots_dir,
                             reward_fname='compare_rewards.png', td_fname='compare_td_error.png',
                             title_prefix=''):
    """Draw and save comparative reward and TD-error histories.

    - `plots_dir` will be created if missing.
    - Returns tuple of saved file paths: (reward_path, td_path).
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Reward comparison
    episodes = range(1, len(rewards_jalgt) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards_jalgt, label='Rewards - JAL-GT')
    plt.plot(episodes, rewards_wolf, label='Rewards - WoLFPHC')
    plt.title(f'{title_prefix} Collective Reward - JAL-GT vs WoLFPHC')
    plt.xlabel('Epoch')
    plt.ylabel('Collective Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    reward_path = os.path.join(plots_dir, reward_fname)
    plt.savefig(reward_path)
    plt.close()

    # TD-error comparison
    episodes_td = range(1, len(td_jalgt) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_td, td_jalgt, label='TD Error - JAL-GT')
    plt.plot(episodes_td, td_wolf, label='TD Error - WoLFPHC')
    plt.title(f'{title_prefix} TD Error - JAL-GT vs WoLFPHC')
    plt.xlabel('Epoch')
    plt.ylabel('TD Error (sum per epoch)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    td_path = os.path.join(plots_dir, td_fname)
    plt.savefig(td_path)
    plt.close()

    return reward_path, td_path






def _save_rewards_and_td(rewards, td_errors, plots_dir, base_name):
    os.makedirs(plots_dir, exist_ok=True)
    # Rewards
    episodes = range(1, len(rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Cumulative Reward')
    plt.title(f'{base_name} - Cumulative Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Collective Reward')
    plt.grid(True)
    plt.tight_layout()
    reward_path = os.path.join(plots_dir, f"{base_name}_rewards.png")
    plt.savefig(reward_path)
    plt.close()

    # TD errors
    episodes_td = range(1, len(td_errors) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_td, td_errors, label='TD Error')
    plt.title(f'{base_name} - TD Error')
    plt.xlabel('Epoch')
    plt.ylabel('TD Error (sum per epoch)')
    plt.grid(True)
    plt.tight_layout()
    td_path = os.path.join(plots_dir, f"{base_name}_td_error.png")
    plt.savefig(td_path)
    plt.close()


def experiment_1():
    """Run WoLFPHC with decreasing amounts of neurons and save plots in plots/experiment0."""
    exp_config = {
        "num_agents": 2,
        "size": 10,
        "maps": 10,
        "num_states": 16 * 16 * 4,
        "epochs": 300,
        "episodes_per_epoch": 10,
        "episode_length": 16,
        "obstacle_density": 0.1,
        "save_every": None,
        "learning_rate": 0.01,
        "epsilon_max": 1,
        "epsilon_min": 0.1,
        "renders": "renders/",
        "algorithm_cls": WoLFPHC,
        "algorithm_kwargs": {},
        "solution_concept": None
    }

    plots_dir = exp_config.get("plots_dir", "plots/experiment0/")
    os.makedirs(plots_dir, exist_ok=True)

    # Define decreasing hidden sizes to test
    hidden_sizes_list = [
        (64, 32)
        (32, 16),
        (16, 8),
        (8, 4)
    ]

    for sizes in hidden_sizes_list:
        exp_config["algorithm_cls"] = WoLFPHC
        exp_config["solution_concept"] = None
        exp_config["algorithm_kwargs"] = {
            "learning_rate_loss": 0.05,
            "learning_rate_win": 0.01,
            "hidden_sizes": list(sizes),
        }

        base_name = f"WoLF_hidden_{sizes[0]}-{sizes[1]}"
        print(f"Running experiment_0 with hidden sizes: {sizes}")
        rewards, td = run_experiment(exp_config)
        _save_rewards_and_td(rewards, td, plots_dir, base_name)



def experiment_2() :
    """Run JAL-GT vs WoLFPHC for board sizes 4, 6 and 10 and save plots per size."""
    sizes_to_test = [4, 6, 10]

    for size in sizes_to_test:
        exp_config = {
            "num_agents": 2,
            "size": size,
            "maps": 10,
            "num_states": 16 * 16 * 4,
            "epochs": 300,
            "episodes_per_epoch": 10,
            "episode_length": 16,
            "obstacle_density": 0.1,
            "save_every": None,
            "learning_rate": 0.01,
            "epsilon_max": 1,
            "epsilon_min": 0.1,
            "renders": "renders/",
            "algorithm_cls": JALGT,
            "algorithm_kwargs": {},
            "solution_concept": ParetoSolutionConcept
        }

        plots_dir = f"plots/experiment1/size_{size}/"
        os.makedirs(plots_dir, exist_ok=True)

        # Run JAL-GT
        exp_config["algorithm_cls"] = JALGT
        exp_config["solution_concept"] = ParetoSolutionConcept
        exp_config["algorithm_kwargs"] = {}
        rewards_jalgt, td_jalgt = run_experiment(exp_config)

        # Run WoLFPHC
        exp_config["algorithm_cls"] = WoLFPHC
        exp_config["solution_concept"] = None
        exp_config["algorithm_kwargs"] = {
            "learning_rate_loss": 0.05,
            "learning_rate_win": 0.01
        }
        rewards_wolf, td_wolf = run_experiment(exp_config)

        # Draw and save comparative plots for this size
        reward_fname = f'experiment2_rewards_size{size}.png'
        td_fname = f'experiment2_td_error_size{size}.png'
        draw_comparative_history(rewards_jalgt, rewards_wolf, td_jalgt, td_wolf,
                                 plots_dir=plots_dir,
                                 reward_fname=reward_fname,
                                 td_fname=td_fname,
                                 title_prefix=f'Size {size}:')
    



if __name__ == '__main__':
    # Run experiment 0 first (WoLFPHC hidden-size sweep), then experiment_1.
    #experiment_1()
    experiment_2()