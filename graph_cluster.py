"""
MEMBER 4 - GRAPH THEORY, RELATIONS & VISUALISATION MODULE
---------------------------------------------------------
Responsibility : Compare ALL documents pairwise, build a "similarity graph",
                 find clusters of plagiarised documents and draw the charts
                 (matplotlib) used in the final report.

Discrete Maths used:
  * Graphs   : vertices = documents, edge = similarity >= threshold
  * Adjacency list, degree, density, Handshaking Lemma  (sum deg = 2|E|)
  * Connected components via BFS
  * Relations: reflexive, symmetric, transitive, equivalence relation
  * Combinations: n documents -> C(n,2) = n(n-1)/2 pairs to compare
  * Sets / Venn diagrams for common vs unique words

Charts produced (saved as PNG, then shown on screen after a short delay):
  01_similarity_graph.png        the graph itself, clusters coloured
  02_metric_comparison.png       Jaccard/Dice/Overlap/Cosine/N-gram/LCS per pair
  03_common_vs_unique_words.png  |A-B|, |A∩B|, |B-A| for every pair
  04_venn_diagram.png            Venn diagram of the most similar pair
"""
import math
import os
import time
from collections import deque
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

METRICS = ["jaccard", "dice", "overlap", "cosine", "ngram", "lcs", "final"]


# =====================================================================
#  PART A : GRAPH THEORY / RELATIONS (pure discrete maths, no plotting)
# =====================================================================
def number_of_pairs(n):
    """C(n,2) = n(n-1)/2 comparisons needed for n documents."""
    return n * (n - 1) // 2


def build_similarity_matrix(names, score_fn):
    """Symmetric n x n matrix; diagonal = 1 (reflexive)."""
    n = len(names)
    S = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i, j in combinations(range(n), 2):
        S[i][j] = S[j][i] = score_fn(i, j)
    return S


def build_graph(names, S, threshold):
    """Adjacency list of the similarity graph (edge if S[i][j] >= threshold)."""
    graph = {name: [] for name in names}
    for i, j in combinations(range(len(names)), 2):
        if S[i][j] >= threshold:
            graph[names[i]].append(names[j])
            graph[names[j]].append(names[i])
    return graph


def vertex_degrees(graph):
    """deg(v) = number of documents that v is suspiciously similar to."""
    return {v: len(nbrs) for v, nbrs in graph.items()}


def connected_components(graph):
    """BFS to find clusters of linked documents."""
    visited, components = set(), []
    for start in graph:
        if start in visited:
            continue
        comp, queue = [], deque([start])
        visited.add(start)
        while queue:
            v = queue.popleft()
            comp.append(v)
            for u in graph[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        components.append(sorted(comp))
    return components


def graph_summary(graph):
    """Vertices, edges, density and a check of the Handshaking Lemma."""
    degrees = vertex_degrees(graph)
    edges = {frozenset((u, w)) for u, nbrs in graph.items() for w in nbrs}
    v, e = len(graph), len(edges)
    return {
        "vertices |V|": v,
        "edges |E|": e,
        "density 2|E|/(|V|(|V|-1))": round(2 * e / (v * (v - 1)), 3) if v > 1 else 0.0,
        "components": len(connected_components(graph)),
        "handshake: sum(deg) == 2|E|": sum(degrees.values()) == 2 * e,
    }


def relation_properties(S, threshold):
    """Check reflexive / symmetric / transitive for 'is similar to'."""
    n = len(S)
    R = [[S[i][j] >= threshold for j in range(n)] for i in range(n)]
    reflexive = all(R[i][i] for i in range(n))
    symmetric = all(R[i][j] == R[j][i] for i in range(n) for j in range(n))
    transitive = all(
        (not (R[i][j] and R[j][k])) or R[i][k]
        for i in range(n) for j in range(n) for k in range(n)
    )
    return {"reflexive": reflexive, "symmetric": symmetric, "transitive": transitive}


def print_matrix(names, S):
    width = max(len(n) for n in names) + 2
    print(" " * width + "".join(f"{n:>{width}}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{width}}" + "".join(f"{S[i][j]:>{width}.2f}" for j in range(len(names))))


def verdict(score):
    if score >= 0.70:
        return "HIGH plagiarism (near copy)"
    if score >= 0.50:
        return "MODERATE similarity (suspicious)"
    if score >= 0.25:
        return "LOW similarity"
    return "Original / unrelated"


# =====================================================================
#  PART B : VISUALISATION (matplotlib)
# =====================================================================
def _label(name):
    return os.path.splitext(name)[0]


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return path


def plot_network(names, S, graph, threshold, out_dir):
    """Chart 2 - the similarity graph. Edge width/colour = score, node size = degree."""
    n = len(names)
    comps = connected_components(graph)
    comp_of = {v: ci for ci, comp in enumerate(comps) for v in comp}
    degrees = vertex_degrees(graph)
    palette = plt.get_cmap("tab10")
    edge_cmap = plt.get_cmap("OrRd")

    # place the members of each cluster next to each other on the circle
    order = [v for comp in sorted(comps, key=len, reverse=True) for v in comp]
    pos = {v: (math.cos(2 * math.pi * k / n + math.pi / 2),
               math.sin(2 * math.pi * k / n + math.pi / 2)) for k, v in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    for i, j in combinations(range(n), 2):
        if S[i][j] >= threshold:
            (x1, y1), (x2, y2) = pos[names[i]], pos[names[j]]
            ax.plot([x1, x2], [y1, y2], color=edge_cmap(0.35 + 0.65 * S[i][j]),
                    lw=1.5 + 8 * S[i][j], solid_capstyle="round", zorder=1)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{S[i][j]:.2f}", ha="center",
                    va="center", fontsize=9, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray"))
    for name in names:
        x, y = pos[name]
        in_cluster = len(comps[comp_of[name]]) > 1
        color = palette(comp_of[name] % 10) if in_cluster else "lightgray"
        ax.scatter([x], [y], s=2600 + 800 * degrees[name], c=[color],
                   edgecolors="black", linewidths=1.6, zorder=2)
        ax.text(x, y, f"{_label(name)}\ndeg={degrees[name]}", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    handles = []
    for ci, comp in enumerate(comps):
        if len(comp) > 1:
            handles.append(Line2D([0], [0], marker="o", color="w", markersize=14,
                                  markerfacecolor=palette(ci % 10), markeredgecolor="black",
                                  label="Cluster: " + ", ".join(_label(c) for c in comp)))
    if any(len(c) == 1 for c in comps):
        handles.append(Line2D([0], [0], marker="o", color="w", markersize=14,
                              markerfacecolor="lightgray", markeredgecolor="black",
                              label="No similar document"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=1, frameon=False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Similarity graph  (edge if score ≥ {threshold};  "
                 "width = score,  node size = degree)", fontweight="bold")
    return _save(fig, out_dir, "01_similarity_graph.png")


def _ranked_pairs(pair_scores, top_k):
    return sorted(pair_scores.items(), key=lambda kv: kv[1]["final"], reverse=True)[:top_k]


def plot_metric_comparison(names, pair_scores, out_dir, top_k=6):
    """Chart 3 - grouped bars: every metric for the most similar pairs."""
    items = _ranked_pairs(pair_scores, top_k)
    labels = [f"{_label(names[i])}\nvs\n{_label(names[j])}" for (i, j), _ in items]
    x = np.arange(len(items))
    w = 0.8 / len(METRICS)
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(METRICS)))

    fig, ax = plt.subplots(figsize=(max(7, 2.0 * len(items) + 3), 6))
    for k, m in enumerate(METRICS):
        offs = (k - (len(METRICS) - 1) / 2) * w
        bars = ax.bar(x + offs, [sc[m] * 100 for _, sc in items], w, label=m,
                      color=colors[k], edgecolor="black" if m == "final" else "none",
                      linewidth=1.5)
        if len(items) <= 4:
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        f"{b.get_height():.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Similarity (%)")
    ax.set_title("Metric comparison for the most similar document pairs", fontweight="bold")
    ax.legend(ncol=len(METRICS), loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "02_metric_comparison.png")


def plot_common_vs_unique(names, docs, pair_scores, out_dir, top_k=10):
    """Chart 4 - for each pair: words only in A | common words | words only in B."""
    items = _ranked_pairs(pair_scores, top_k)
    labels, only_a, common, only_b = [], [], [], []
    for (i, j), _ in items:
        a, b = set(docs[i]["keywords"]), set(docs[j]["keywords"])
        labels.append(f"{_label(names[i])} vs {_label(names[j])}")
        only_a.append(len(a - b))
        common.append(len(a & b))
        only_b.append(len(b - a))

    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(9, 1.0 * len(items) + 2.5))
    ax.barh(y, only_a, color="#6baed6", edgecolor="black", label="Only in first  (A − B)")
    ax.barh(y, common, left=only_a, color="#fd8d3c", edgecolor="black", label="Common  (A ∩ B)")
    ax.barh(y, only_b, left=np.add(only_a, common), color="#74c476", edgecolor="black",
            label="Only in second  (B − A)")
    for k in range(len(items)):
        for val, start in ((only_a[k], 0), (common[k], only_a[k]), (only_b[k], only_a[k] + common[k])):
            if val:
                ax.text(start + val / 2, k, str(val), ha="center", va="center", fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of unique keywords")
    ax.set_title("Common vs unique words per pair", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, out_dir, "03_common_vs_unique_words.png")


def plot_venn(name_a, name_b, set_a, set_b, out_dir):
    """Chart 5 - Venn diagram of the keyword sets (circles drawn closer = more similar)."""
    set_a, set_b = set(set_a), set(set_b)
    inter, union = len(set_a & set_b), len(set_a | set_b)
    jac = inter / union if union else 1.0
    r = 1.0
    d = 0.5 + 1.3 * (1 - jac)          # distance between centres

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.add_patch(Circle((-d / 2, 0), r, color="#6baed6", alpha=0.55, ec="black", lw=1.5))
    ax.add_patch(Circle((d / 2, 0), r, color="#74c476", alpha=0.55, ec="black", lw=1.5))
    ax.text(-r, 0.12, str(len(set_a - set_b)), ha="center", fontsize=20, fontweight="bold")
    ax.text(-r, -0.2, "only A", ha="center", fontsize=10)
    ax.text(0, 0.12, str(inter), ha="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.2, "A ∩ B", ha="center", fontsize=10)
    ax.text(r, 0.12, str(len(set_b - set_a)), ha="center", fontsize=20, fontweight="bold")
    ax.text(r, -0.2, "only B", ha="center", fontsize=10)
    ax.text(-d / 2, r + 0.15, _label(name_a), ha="center", fontsize=12, fontweight="bold")
    ax.text(d / 2, r + 0.15, _label(name_b), ha="center", fontsize=12, fontweight="bold")
    ax.text(0, -r - 0.45,
            f"Jaccard = |A∩B| / |A∪B| = {inter} / {union} = {jac:.2f}\n"
            f"|A∪B| = |A| + |B| − |A∩B| = {len(set_a)} + {len(set_b)} − {inter} = {union}",
            ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="gray"))
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.2, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Venn diagram of keyword sets (most similar pair)", fontweight="bold")
    return _save(fig, out_dir, "04_venn_diagram.png")


def generate_all_plots(names, docs, S, graph, threshold, pair_scores, out_dir="output"):
    """Draw and save every chart. The Venn diagram uses the most similar pair."""
    saved = [
        plot_network(names, S, graph, threshold, out_dir),
        plot_metric_comparison(names, pair_scores, out_dir),
        plot_common_vs_unique(names, docs, pair_scores, out_dir),
    ]
    i, j = max(pair_scores, key=lambda k: pair_scores[k]["final"])
    saved.append(plot_venn(names[i], names[j], docs[i]["keywords"], docs[j]["keywords"], out_dir))
    return saved


def show_plots(delay=5):
    """Wait `delay` seconds (with a countdown), then pop every chart up on screen."""
    for remaining in range(int(delay), 0, -1):
        print(f"  Charts will open in {remaining:2d} s ...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")
    plt.show()          # one window per chart; close them to finish
    plt.close("all")


def close_plots():
    plt.close("all")


if __name__ == "__main__":
    names = ["A", "B", "C", "D"]
    S = [[1, .9, .1, 0], [.9, 1, .6, 0], [.1, .6, 1, 0], [0, 0, 0, 1]]
    S = [[float(x) for x in r] for r in S]
    g = build_graph(names, S, 0.5)
    print("Graph      :", g)
    print("Degrees    :", vertex_degrees(g))
    print("Components :", connected_components(g))
    print("Summary    :", graph_summary(g))
    print("Relation   :", relation_properties(S, 0.5))
    print("Pairs C(4,2) =", number_of_pairs(4))
    print("Saved      :", plot_network(names, S, g, 0.5, "output"))
    close_plots()
