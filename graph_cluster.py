"""GRAPH MODULE - similarity graph, clusters and charts."""
import math
import os
import time
from collections import deque
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

METRICS = ["jaccard", "dice", "overlap", "cosine", "ngram", "lcs", "final"]


# ---------- PART A: graph theory ----------

def number_of_pairs(n):
    """C(n,2) = n(n-1)/2 comparisons for n documents."""
    return n * (n - 1) // 2


def build_similarity_matrix(names, score_fn):
    """n x n table of scores (diagonal = 1, same score both ways)."""
    n = len(names)
    S = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i, j in combinations(range(n), 2):
        S[i][j] = S[j][i] = score_fn(i, j)
    return S


def build_graph(names, S, threshold):
    """Graph as a dictionary: document -> list of similar documents."""
    graph = {name: [] for name in names}
    for i, j in combinations(range(len(names)), 2):
        if S[i][j] >= threshold:
            graph[names[i]].append(names[j])
            graph[names[j]].append(names[i])
    return graph


def vertex_degrees(graph):
    """Degree = how many documents each document is similar to."""
    return {v: len(neighbours) for v, neighbours in graph.items()}


def connected_components(graph):
    """Find groups (clusters) of linked documents using BFS."""
    visited, groups = set(), []
    for start in graph:
        if start in visited:
            continue
        group, queue = [], deque([start])
        visited.add(start)
        while queue:
            v = queue.popleft()
            group.append(v)
            for u in graph[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        groups.append(sorted(group))
    return groups


def print_matrix(names, S):
    """Print the similarity table."""
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{n:>{w}}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{S[i][j]:>{w}.2f}" for j in range(len(names))))


def verdict(score):
    """Turn a score into a simple message."""
    if score >= 0.70:
        return "HIGH plagiarism (near copy)"
    if score >= 0.50:
        return "MODERATE similarity (suspicious)"
    if score >= 0.25:
        return "LOW similarity"
    return "Original / unrelated"


# ---------- PART B: charts ----------

def _label(name):
    return os.path.splitext(name)[0]          # "doc1.txt" -> "doc1"


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    return path


def _top_pairs(pair_scores, k):
    """The k most similar pairs."""
    return sorted(pair_scores.items(), key=lambda p: p[1]["final"], reverse=True)[:k]


def plot_network(names, S, graph, threshold, out_dir):
    """Chart 1: documents as circles, lines between similar ones."""
    n = len(names)
    colors = {}
    for c, group in enumerate(connected_components(graph)):
        for v in group:
            colors[v] = plt.get_cmap("tab10")(c % 10) if len(group) > 1 else "lightgray"

    # place the documents evenly on a circle
    pos = {name: (math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
           for k, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, j in combinations(range(n), 2):
        if S[i][j] >= threshold:
            (x1, y1), (x2, y2) = pos[names[i]], pos[names[j]]
            ax.plot([x1, x2], [y1, y2], color="red", lw=1 + 6 * S[i][j], zorder=1)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{S[i][j]:.2f}", ha="center",
                    fontsize=9, zorder=3, bbox=dict(fc="white", ec="gray"))
    for name in names:
        x, y = pos[name]
        ax.scatter([x], [y], s=2500, c=[colors[name]], edgecolors="black", zorder=2)
        ax.text(x, y, _label(name), ha="center", va="center", fontweight="bold", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Similarity graph (line if score >= {threshold})", fontweight="bold")
    return _save(fig, out_dir, "01_similarity_graph.png")


def plot_metric_comparison(names, pair_scores, out_dir):
    """Chart 2: bars for every metric, for the most similar pairs."""
    items = _top_pairs(pair_scores, 6)
    labels = [f"{_label(names[i])}\nvs\n{_label(names[j])}" for (i, j), _ in items]
    x = np.arange(len(items))
    width = 0.8 / len(METRICS)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k, metric in enumerate(METRICS):
        offset = (k - (len(METRICS) - 1) / 2) * width
        ax.bar(x + offset, [sc[metric] * 100 for _, sc in items], width, label=metric)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Similarity (%)")
    ax.set_title("Metric comparison", fontweight="bold")
    ax.legend(ncol=len(METRICS), loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    return _save(fig, out_dir, "02_metric_comparison.png")


def plot_common_vs_unique(names, docs, pair_scores, out_dir):
    """Chart 3: words only in A | common words | words only in B."""
    items = _top_pairs(pair_scores, 10)
    labels, only_a, common, only_b = [], [], [], []
    for (i, j), _ in items:
        a, b = set(docs[i]["keywords"]), set(docs[j]["keywords"])
        labels.append(f"{_label(names[i])} vs {_label(names[j])}")
        only_a.append(len(a - b))       # set difference
        common.append(len(a & b))       # intersection
        only_b.append(len(b - a))

    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(9, len(items) + 2.5))
    ax.barh(y, only_a, color="#6baed6", label="Only in first (A - B)")
    ax.barh(y, common, left=only_a, color="#fd8d3c", label="Common (A & B)")
    ax.barh(y, only_b, left=np.add(only_a, common), color="#74c476", label="Only in second (B - A)")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of unique keywords")
    ax.set_title("Common vs unique words", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, out_dir, "03_common_vs_unique_words.png")


def plot_venn(name_a, name_b, set_a, set_b, out_dir):
    """Chart 4: Venn diagram of the keywords of the most similar pair."""
    a, b = set(set_a), set(set_b)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_patch(Circle((-0.5, 0), 1, color="#6baed6", alpha=0.55, ec="black"))
    ax.add_patch(Circle((0.5, 0), 1, color="#74c476", alpha=0.55, ec="black"))
    ax.text(-1, 0, f"{len(a - b)}\nonly A", ha="center", va="center", fontsize=14)
    ax.text(0, 0, f"{len(a & b)}\nboth", ha="center", va="center", fontsize=14)
    ax.text(1, 0, f"{len(b - a)}\nonly B", ha="center", va="center", fontsize=14)
    ax.text(-0.5, 1.15, _label(name_a), ha="center", fontweight="bold")
    ax.text(0.5, 1.15, _label(name_b), ha="center", fontweight="bold")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.3, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Venn diagram (most similar pair)", fontweight="bold")
    return _save(fig, out_dir, "04_venn_diagram.png")


def generate_all_plots(names, docs, S, graph, threshold, pair_scores, out_dir="output"):
    """Draw and save all 4 charts."""
    i, j = max(pair_scores, key=lambda k: pair_scores[k]["final"])
    return [
        plot_network(names, S, graph, threshold, out_dir),
        plot_metric_comparison(names, pair_scores, out_dir),
        plot_common_vs_unique(names, docs, pair_scores, out_dir),
        plot_venn(names[i], names[j], docs[i]["keywords"], docs[j]["keywords"], out_dir),
    ]


def show_plots(delay=5):
    """Wait a few seconds, then open the chart windows."""
    time.sleep(delay)
    plt.show()
    plt.close("all")


def close_plots():
    plt.close("all")