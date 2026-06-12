from collections import deque
from pogema import pogema_v0, GridConfig

N_SEEDS = 30


def reachable(obstacles, start, goal):
    if start == goal:
        return True
    h, w = obstacles.shape
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in seen and not obstacles[nx][ny]:
                if (nx, ny) == goal:
                    return True
                seen.add((nx, ny))
                queue.append((nx, ny))
    return False


for size in (4, 6, 10):
    for density in (0.1, 0.3):
        ok = 0
        for seed in range(N_SEEDS):
            cfg = GridConfig(num_agents=2, size=size, density=density, seed=seed,
                             max_episode_steps=4 * size, obs_radius=1, on_target="finish")
            env = pogema_v0(cfg)
            env.reset()
            grid = env.unwrapped.grid  # si da error, probar: print(dir(env.unwrapped))
            obstacles, starts, goals = grid.obstacles, grid.positions_xy, grid.finishes_xy
            if all(reachable(obstacles, tuple(starts[i]), tuple(goals[i])) for i in range(2)):
                ok += 1
        print(f"size={size} density={density}: {ok}/{N_SEEDS} mapas resolubles")