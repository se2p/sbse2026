"""Plotting details kept off the algorithm slides."""
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_hypervolume(points, reference):
    from sbse_support import hypervolume_2d
    fig, ax = plt.subplots(figsize=(5, 4))
    lowest = reference[1]
    for x, y in sorted(set(points)):
        if x <= reference[0] and y < lowest:
            ax.add_patch(Rectangle((x, y), reference[0] - x, lowest - y,
                                  facecolor="tab:blue", alpha=0.25, edgecolor="black"))
            lowest = y
    if points:
        ax.scatter(*zip(*points), color="tab:blue", label="minimisation front")
    ax.scatter(*reference, marker="x", color="tab:red", label="reference")
    ax.set(xlabel="Objective 1 (minimise)", ylabel="Objective 2 (minimise)",
           title=f"Non-overlapping dominated rectangles: HV = {hypervolume_2d(points, reference):.2f}")
    ax.autoscale_view()
    ax.legend()
    plt.show()


def plot_goal_trace(baseline, seeded, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, trace, title in zip(axes, [baseline, seeded], ["MOSA", "MOSA + LLM seeds"]):
        ax.imshow(np.array([row["covered"] for row in trace]).T, aspect="auto",
                  cmap="binary", interpolation="nearest", origin="upper")
        ax.set(title=title, xlabel="Generation")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.tight_layout()
    plt.show()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for trace, label in [(baseline, "MOSA"), (seeded, "MOSA + LLM")]:
        x = [row["evaluations"] for row in trace]
        axes[0].plot(x, [sum(row["covered"]) for row in trace], label=label)
        axes[1].plot(x, [row["goal_distances"][-2] for row in trace], label=label)
    for row in seeded:
        if row["seeded"]:
            axes[0].axvline(row["evaluations"], linestyle="--", color="tab:green", label="seed evaluated")
    axes[0].set_ylabel("Goals covered")
    axes[0].legend()
    axes[1].set(xlabel="Test executions", ylabel="Best distance to magic string")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def plot_gantt(panels, jobs, dispatch, num_machines):
    rng = random.Random(7)
    failures = {job.id: rng.random() < job.fail_prob for job in jobs}
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True, squeeze=False)
    colors = plt.get_cmap("tab20")
    for ax, (label, code, result) in zip(axes[:, 0], panels):
        for job, machine, start, end in dispatch(code, jobs):
            failed = failures[job.id]
            ax.barh(machine, end - start, left=start, height=0.8, color=colors(job.id % 20),
                    edgecolor="red" if failed else "none", linewidth=2.5 if failed else 0,
                    hatch="//" if failed else None)
        ax.set_yticks(range(num_machines), [f"machine {i}" for i in range(num_machines)])
        ax.set_title(f"{label}: fitness={result['fitness']:.2f}, makespan={result['makespan']:.1f}")
    axes[-1, 0].set_xlabel("Time")
    fig.suptitle("Same jobs and sampled outcomes; red outlines indicate failures")
    fig.tight_layout()
    plt.show()


def plot_partition_encodings():
    parent = [0, 0, 1, 1, 2, 2]
    equivalent = [2, 2, 0, 0, 1, 1]
    other = [0, 1, 0, 1, 2, 2]
    rows = [parent, equivalent, other, parent[:3] + other[3:], equivalent[:3] + other[3:]]
    labels = ["Parent A", "Same partition, different labels", "Parent B",
              "A × B", "Relabelled A × B"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(rows, cmap="Set2", vmin=0, vmax=2, aspect="auto")
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks(range(6), [f"module {i}" for i in range(6)])
    ax.axvline(2.5, color="black", linestyle="--")
    ax.set_title("Cluster labels are arbitrary; single-point crossover is label-sensitive")
    fig.tight_layout()
    plt.show()
