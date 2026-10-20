"""Small, non-algorithmic helpers shared by the lecture notebooks.

Search loops and operators stay in the notebooks. This module handles recording,
replay and interruptible execution so those details do not fill RISE slides.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

FULL_EXPERIMENTS = os.environ.get("SBSE_FULL_EXPERIMENTS", "0") == "1"
REPETITIONS = 30 if FULL_EXPERIMENTS else 5


def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


class BudgetExhausted(Exception):
    pass


class FitnessRun:
    """Cache fitness on each unchanged candidate and count actual objective work.

    These early notebooks deliberately use a global problem definition. The
    context restores it on exit. New candidate objects are evaluated once;
    selection and survivor comparisons reuse their recorded score. For suites,
    cost=len counts individual test executions rather than suite evaluations.
    """

    def __init__(self, namespace, budget, maximize=True, cost=lambda _: 1):
        if budget < 1:
            raise ValueError("The evaluation budget must be positive")
        self.namespace, self.budget, self.maximize, self.cost = namespace, budget, maximize, cost
        self.history, self.cache = [], {}
        self.best, self.best_fitness = None, -math.inf if maximize else math.inf
        self.used = self.calls = 0

    @property
    def remaining(self):
        return self.budget - self.used

    def __enter__(self):
        self.original = self.namespace["get_fitness"]
        self.namespace["get_fitness"] = self.evaluate
        self.namespace["fitness_values"] = self.history
        return self

    def evaluate(self, candidate):
        import copy
        cached = self.cache.get(id(candidate))
        if cached is not None and _unchanged(cached[1], candidate):
            return cached[2]
        work = self.cost(candidate)
        if work > self.remaining or not self.remaining:
            raise BudgetExhausted()
        score = self.original(candidate)
        self.cache[id(candidate)] = (candidate, copy.deepcopy(candidate), score)
        self.used += work
        self.calls += 1
        if self.best is None or (score > self.best_fitness if self.maximize else score < self.best_fitness):
            self.best, self.best_fitness = copy.deepcopy(candidate), score
        self.history.extend([self.best_fitness] * work)
        return score

    def __exit__(self, kind, value, traceback):
        self.namespace["get_fitness"] = self.original
        return kind is BudgetExhausted


def _unchanged(snapshot, candidate):
    """True if a cached deep copy still equals the candidate (lists, arrays, plain objects)."""
    if type(candidate).__eq__ is object.__eq__ and hasattr(candidate, "__dict__"):
        return type(snapshot) is type(candidate) and vars(snapshot) == vars(candidate)
    equal = snapshot == candidate
    return bool(equal.all()) if hasattr(equal, "all") else bool(equal)


class Evaluations:
    """Count every objective call, including initialization and repeated candidates."""

    def __init__(self, fitness, budget):
        if budget < 1:
            raise ValueError("The evaluation budget must be positive")
        self.fitness, self.budget = fitness, budget
        self.best, self.best_fitness = None, -math.inf
        self.history, self.current_history = [], []

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def __call__(self, candidate):
        if not self.remaining:
            raise RuntimeError("Evaluation budget exhausted")
        score = self.fitness(candidate)
        if score > self.best_fitness:
            import copy
            self.best, self.best_fitness = copy.deepcopy(candidate), score
        self.history.append(self.best_fitness)
        return score

    def result(self):
        return self.best, self.history


def hypervolume_2d(points, reference):
    """Area dominated by minimisation points and bounded by a dominated reference.

    Duplicates and dominated points are allowed; the input is never modified.
    Points outside the reference box contribute zero, rather than an abs() area.
    """
    rx, ry = reference
    area, lowest = 0.0, ry
    for x, y in sorted(set(tuple(p) for p in points)):
        if x <= rx and y < lowest:
            area += (rx - x) * (lowest - y)
            lowest = y
    return area


def crowding_sort(front, objectives=("fitness1", "fitness2")):
    """Assign NSGA-II crowding distance, skipping constant objectives."""
    for individual in front:
        individual.distance = 0.0
    for objective in objectives:
        ordered = sorted(front, key=lambda x: getattr(x, objective))
        if not ordered:
            continue
        span = getattr(ordered[-1], objective) - getattr(ordered[0], objective)
        if span == 0:
            continue
        ordered[0].distance = ordered[-1].distance = math.inf
        for previous, current, following in zip(ordered, ordered[1:], ordered[2:]):
            current.distance += (getattr(following, objective) - getattr(previous, objective)) / span
    front.sort(key=lambda x: x.distance, reverse=True)


def pair_entropy(symbols, same=False):
    """Entropy of overlapping symbol pairs; an empty pair set has entropy zero."""
    counts = Counter(zip(symbols, symbols[1:]))
    counts = {pair: count for pair, count in counts.items() if (pair[0] == pair[1]) == same}
    # H(epsilon) uses all pairs as its denominator. Basin information is the
    # entropy conditional on a same-symbol pair, so a constant plateau has H=0.
    total = sum(counts.values()) if same else max(0, len(symbols) - 1)
    base = 3 if same else 6
    return -sum((n / total) * math.log(n / total, base) for n in counts.values()) if total else 0.0


def holm(pvalues):
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, (index, p) in enumerate(sorted(enumerate(pvalues), key=lambda x: x[1])):
        previous = max(previous, (len(pvalues) - rank) * p)
        adjusted[index] = min(1.0, previous)
    return adjusted


def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X = Y) for independent samples."""
    greater = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    return greater / (len(x) * len(y))


def compare(samples, pairs=None, higher_is_better=True, resamples=2000, seed=0):
    """Mann-Whitney U tests with Holm correction and A12 with a bootstrap 95% interval.

    samples maps a label to final results (one value per run); pairs defaults to
    all label pairs. A12 > 0.5 means the first label tends to be better. The
    interval resamples each group independently; with five runs it is very wide.
    """
    import itertools
    import random
    from scipy.stats import mannwhitneyu
    pairs = list(pairs or itertools.combinations(samples, 2))
    sign = 1 if higher_is_better else -1
    rng, rows = random.Random(seed), []
    for first, second in pairs:
        x = [sign * v for v in samples[first]]
        y = [sign * v for v in samples[second]]
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        boot = sorted(a12(rng.choices(x, k=len(x)), rng.choices(y, k=len(y))) for _ in range(resamples))
        rows.append({"first": first, "second": second, "n": (len(x), len(y)), "p": p,
                     "A12": a12(x, y), "A12_low": boot[int(0.025 * resamples)],
                     "A12_high": boot[int(0.975 * resamples) - 1]})
    for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = adjusted
    return rows


def print_comparison(rows, alpha=0.05):
    """Print compare() results as a compact table."""
    def fmt(p):
        return f"{p:8.4f}" if p >= 1e-4 else f"{p:8.0e}"
    width = max(len(f"{row['first']} vs {row['second']}") for row in rows) + 2
    print(f"{'comparison':{width}} {'n':>5} {'p':>8} {'Holm p':>8}  A12 [95% bootstrap]")
    for row in rows:
        name = f"{row['first']} vs {row['second']}"
        verdict = "significant" if row["p_holm"] < alpha else "not significant"
        print(f"{name:{width}} {row['n'][0]:>2}/{row['n'][1]:<2} {fmt(row['p'])} {fmt(row['p_holm'])}  "
              f"{row['A12']:.2f} [{row['A12_low']:.2f}, {row['A12_high']:.2f}]   {verdict}")


def extract_code(text):
    import re
    # The optional language tag (python, py, Python3, ...) is not part of the code.
    match = re.search(r"```[\w+.-]*[ \t]*\n?(.*?)```", text, re.S)
    return match.group(1).strip() if match else text.strip()


def llm_client(cache_file, revision="1"):
    """Return an ask(prompt) function. Replay never invokes an endpoint or CLI.

    The shipped prompt-only entries are legacy teaching recordings with unknown
    model provenance. New live entries include their complete request settings.
    Live mode intentionally bypasses recordings, even if the prompt is cached.
    """
    path = Path(cache_file)
    cache = json.loads(path.read_text()) if path.exists() else {}
    mode = os.environ.get("LLM_MODE", "replay")
    if mode not in {"replay", "live"}:
        raise ValueError("LLM_MODE must be replay or live")
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    # The CLI accepts no temperature, so CLI recordings must not claim one.
    settings = {"model": os.environ.get("LLM_MODEL") or ("sonnet" if not base else ""),
                "base_url": base, "temperature": 0 if base else None, "prompt_revision": revision}

    def ask(prompt):
        metadata = {**settings, "prompt": prompt}
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        legacy = hashlib.sha256(prompt.encode()).hexdigest()
        if mode == "replay":
            entry = cache.get(key, cache.get(legacy))
            if entry is None:
                raise KeyError("No recorded answer for this prompt. Use LLM_MODE=live explicitly to generate one.")
            return entry if isinstance(entry, str) else entry["answer"]
        if base:
            import requests
            if not settings["model"]:
                raise ValueError("Set LLM_MODEL when using an endpoint")
            response = requests.post(base + "/chat/completions", timeout=60,
                headers={"Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "")},
                json={"model": settings["model"], "temperature": settings["temperature"],
                      "messages": [{"role": "user", "content": prompt}]})
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        else:
            answer = subprocess.run(["claude", "-p", "--model", settings["model"], prompt],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=60, check=True).stdout
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The model returned an empty or invalid answer")
        from datetime import datetime, timezone
        cache[key] = {"answer": answer, "request": metadata,
                      "recorded_at": datetime.now(timezone.utc).isoformat()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2) + "\n")
        return answer

    def can_replay(prompt):
        key = hashlib.sha256(json.dumps({**settings, "prompt": prompt}, sort_keys=True).encode()).hexdigest()
        legacy = hashlib.sha256(prompt.encode()).hexdigest()
        return key in cache or legacy in cache

    ask.can_replay = can_replay
    ask.mode = mode
    return ask


def isolated_python(source, expression, bindings=None, timeout=2.0):
    """Execute a teaching candidate in a killable process (not a security sandbox)."""
    request = {"source": source, "expression": expression, "bindings": bindings or {}}
    worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(request), capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(worker.stdout)


def _worker():
    import contextlib
    import io
    import random
    import time
    from types import SimpleNamespace
    request = json.load(sys.stdin)
    namespace = {"random": random.Random(0), "time": time, "SimpleNamespace": SimpleNamespace,
                 "math": math, **request["bindings"]}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(request["source"], namespace)
        result = eval(request["expression"], namespace)
    print(json.dumps(result, default=vars, allow_nan=False))


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
