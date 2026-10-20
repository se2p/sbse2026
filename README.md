# Search-Based Software Engineering

Lecture notebooks for the course on Search-Based Software Engineering. Chapters
are added here once they have been presented in class. Every notebook is also
available as rendered Markdown in [rendered/](rendered/), which you can read
without running anything.

## Chapters

| Week | Core lecture path | Notebook |
|---|---|---|
| 1 | Encoding, objective, random search, neighbourhoods, hill climbing; compiler flags | [Random and Local Search – Part 1](Random%20and%20Local%20Search%20-%20Part%201.ipynb) ([rendered](rendered/Random%20and%20Local%20Search%20-%20Part%201.md)) |
| 2 | Restarts, tabu search and simulated annealing (Part 1); fair budgets, distributions and effect sizes; measuring fitness landscapes (Part 2) | [Random and Local Search – Part 1](Random%20and%20Local%20Search%20-%20Part%201.ipynb) ([rendered](rendered/Random%20and%20Local%20Search%20-%20Part%201.md)) |

## Running the notebooks

Use Python 3.12. Some chapters draw trees and networks, which also requires
Graphviz's `dot` executable; install Graphviz with your operating system's
package manager.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name sbse2026 --display-name "SBSE 2026"
jupyter notebook
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Select the **SBSE 2026** kernel, run from the repository root, and use
**Restart & Run All**.
