# SID práctica 3

Naiara González, Gerard Félez, Yaoyao Guo

---

# Bloque 1 — Comparativa algorítmica (JAL-GT vs IQL)

Este bloque compara JAL-GT (baseline, con Pareto) frente a IQL en POGEMA, en las 12 configuraciones obligatorias (entorno, gamma, epsilon). 

## Estructura

```
src/
├── algorithms.py           # JAL-GT y WoLF-PHC
├── iql.py                  # IQL
├── game_model.py           # modelo de juego y espacio de acciones conjuntas
├── solution_concepts.py    # Pareto, Nash, Minimax, Welfare
├── utils.py                # draw_history
├── main.py                 # baseline original
├── main1.py                # bloque 1: JAL-GT vs IQL
├── main2_soluciones.py     # bloque 2: comparativa de conceptos de solución
├── main_JAL-GT_vs_WolfPHC.py  # ampliación
├── check_density.py        # verificación BFS de mapas resolubles
├── requirements.txt
├── results1/               # CSVs y figuras del bloque 1
├── plots/                  # figuras del bloque 2
└── plotsExtra1/            # figuras de la ampliación
```

## Instalación

```bash
python -m venv venv && source venv/bin/activate 
pip install -r requirements.txt
```

Requiere Python 3.10 u 3.11 (pogema 1.2.2 fija gymnasium 0.28.1, que no compila en 3.12).

## Cómo reproducir los experimentos

`main1.py` lanza una configuración: entrena JAL-GT e IQL (3 semillas cada uno)
con los parámetros de `exp_config`, y guarda CSV + figuras + animaciones.

Para cada una de las 12 filas de la tabla siguiente:

1. Abrir `main1.py`.
2. En `exp_config`, cambiar `"tag"`, `"size"`, `"obstacle_density"`, `"episode_length"`,
   `"gamma"` y `"epsilon_min"` por los valores de la fila.
3. Ejecutar:
   ```bash
   python main1.py
   ```
4. Repetir con la siguiente fila.

El resto de campos de `exp_config` (num_agents, maps, epochs, episodes_per_epoch,
learning_rate, epsilon_max, solution_concept, seeds, save_every, renders) no cambian entre configuraciones.

### Tabla de las 12 configuraciones

`episode_length = 4 * size` en todas.

| # | tag | size | obstacle_density | episode_length | gamma | epsilon_min |
|---|-----|------|------------------|----------------|-------|-------------|
| 1 | env_4x4_low | 4 | 0.1 | 16 | 0.95 | 0.1 |
| 2 | env_4x4_med | 4 | 0.3 | 16 | 0.95 | 0.1 |
| 3 | env_6x6_low | 6 | 0.1 | 24 | 0.95 | 0.1 |
| 4 | env_6x6_med | 6 | 0.3 | 24 | 0.95 | 0.1 |
| 5 | env_10x10_low | 10 | 0.1 | 40 | 0.95 | 0.1 |
| 6 | env_10x10_med | 10 | 0.3 | 40 | 0.95 | 0.1 |
| 7 | gamma_0.5 | 6 | 0.3 | 24 | 0.5 | 0.1 |
| 8 | gamma_0.9 | 6 | 0.3 | 24 | 0.9 | 0.1 |
| 9 | gamma_0.99 | 6 | 0.3 | 24 | 0.99 | 0.1 |
| 10 | eps_0.01 | 6 | 0.3 | 24 | 0.95 | 0.01 |
| 11 | eps_0.1 | 6 | 0.3 | 24 | 0.95 | 0.1 |
| 12 | eps_0.3 | 6 | 0.3 | 24 | 0.95 | 0.3 |

**Nota:** la fila 11 (`eps_0.1`) tiene los mismos parámetros que la fila 4
(`env_6x6_med`).

## Salida (por cada `tag` ejecutado)

- `results1/<tag>_summary.csv`: recompensa final, éxito final y tiempo, por algoritmo.
- `results1/figures/<tag>_reward.png`, `<tag>_success.png`, `<tag>_td.png`: curvas con bandas media ± std entre semillas.
- `results1/renders/<tag>_JALGT-agent{0,1}.svg`, `<tag>_IQL-agent{0,1}.svg`: animaciones de evaluación (semilla 0), para el análisis cualitativo.

## Verificación de densidades

`check_density.py` comprueba con BFS si los mapas generados son resolubles, y es la base de la justificación de las densidades 0.1/0.3 (ver documentación).

---

# Bloque 2 — Comparativa de conceptos de solución en JAL-GT

Este bloque compara los distintos conceptos de solución disponibles en JAL-GT (Pareto, Nash, Minimax, Welfare) sobre el mismo entorno POGEMA.

## Cómo reproducir los experimentos

`main2_soluciones.py` lanza **una** configuración con **un** concepto de solución.

Para reproducir la comparativa completa:

1. Abrir `main2_soluciones.py`.
2. En `exp_config`, cambiar `"solution_concept"` por el concepto a probar:
   ```python
   "solution_concept": ParetoSolutionConcept    # o Nash, Minimax, Welfare
   ```
3. Ajustar también `"renders"` para que cada concepto guarde en su propio directorio
   (evita sobreescribir animaciones):
   ```python
   "renders": "renders/pareto/"    # o nash/, minimax/, welfare/
   ```
4. Ejecutar:
   ```bash
   python main2_soluciones.py
   ```
5. Repetir para cada concepto.

Los conceptos disponibles (importados de `solution_concepts.py`):

| Concepto de solución | Clase |
|---|---|
| Óptimo de Pareto | `ParetoSolutionConcept` |
| Equilibrio de Nash | `NashSolutionConcept` |
| Minimax | `MinimaxSolutionConcept` |
| Welfare (bienestar) | `WelfareSolutionConcept` |

### Configuración base para el bloque 2

```python
exp_config = {
    "num_agents": 2,
    "size": 4,
    "maps": 10,
    "epochs": 200,
    "episodes_per_epoch": 10,
    "episode_length": 16,
    "obstacle_density": 0.1,
    "learning_rate": 0.01,
    "epsilon_max": 1,
    "epsilon_min": 0.1,
    "solution_concept": ParetoSolutionConcept,  # cambiar aquí
    "renders": "renders/",
    "save_every": None,
}
```

## Salida

- Gráficas de recompensa colectiva y TD error por epoch (mostradas en pantalla y
  guardadas en `plots/`).
- Animaciones SVG en `renders/<concepto>/` para el análisis cualitativo.
- Métricas por consola: recompensa total por agente, recompensa por mapa en el
  último epoch, tasa de objetivos completados.

---

# Ampliación — JAL-GT vs WoLF-PHC

Como extensión, comparamos JAL-GT (Pareto) frente a WoLF-PHC sobre distintos
tamaños de entorno y densidades de obstáculos. El punto de entrada es
`main_JAL-GT_vs_WolfPHC.py`.

```bash
python main_JAL-GT_vs_WolfPHC.py
```

Los resultados se guardan en `plotsExtra1/`. La justificación y análisis están en
la documentación de ampliación.
