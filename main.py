"""MAIN - runs all four modules.  Usage: python main.py"""
import os

from preprocessing import read_file, preprocess, tokenize
from set_similarity import jaccard, dice, overlap, cosine_multiset
from lcs_module import (lcs_similarity, lcs_alignment, longest_common_substring,
                        sentence_level_matches, changed_words)
from graph_cluster import (build_similarity_matrix, build_graph, vertex_degrees,
                           connected_components, print_matrix, number_of_pairs,
                           verdict, generate_all_plots, show_plots, close_plots)

# ---------- Settings (change these) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE_DIR, "sample_docs")   # folder with .txt files
OUT_DIR = os.path.join(BASE_DIR, "output")       # charts are saved here
THRESHOLD = 0.5                                  # score needed to call a pair similar
SHOW_CHARTS = True                               # False = only save the PNGs

# Weights for the final score (they add up to 1)
WEIGHTS = {"jaccard": 0.20, "ngram": 0.25, "cosine": 0.15, "lcs": 0.40}

SAMPLE_DOCS = {
    "doc1.txt": "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning and self correction. Modern systems use large amounts of data to improve their performance over time. Many industries such as healthcare and finance now depend on these tools.",
    "doc4.txt": "The monsoon season in Mumbai brings heavy rainfall from June to September. Local trains often get delayed and streets fill with water. Street vendors sell hot tea and corn to people waiting for the rain to stop.",
    "doc6.txt": "Heavy rainfall arrives in Mumbai during the monsoon season from June to September. Local trains often get delayed and streets fill with water. Vendors on the street sell hot tea and corn to people waiting for the rain to stop.",
}


def get_paths():
    """Return the .txt files in FOLDER (create demo files if there are none)."""
    os.makedirs(FOLDER, exist_ok=True)
    if len([f for f in os.listdir(FOLDER) if f.endswith(".txt")]) < 2:
        for name, text in SAMPLE_DOCS.items():
            with open(os.path.join(FOLDER, name), "w", encoding="utf-8") as f:
                f.write(text)
    return sorted(os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.endswith(".txt"))


def analyze_pair(a, b):
    """Calculate all similarity scores for two documents."""
    scores = {
        "jaccard": jaccard(a["keywords"], b["keywords"]),
        "dice": dice(a["keywords"], b["keywords"]),
        "overlap": overlap(a["keywords"], b["keywords"]),
        "cosine": cosine_multiset(a["keywords"], b["keywords"]),
        "ngram": jaccard(a["ngrams"], b["ngrams"]),
        "lcs": lcs_similarity(a["tokens"], b["tokens"]),
    }
    scores["final"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return scores


def report(name_a, name_b, a, b, scores):
    """Print the result for one pair of documents."""
    print("=" * 60)
    print(f"{name_a} vs {name_b}")
    print("=" * 60)
    for k in ("jaccard", "dice", "overlap", "cosine", "ngram", "lcs"):
        print(f"  {k:<8}: {scores[k] * 100:6.2f} %")
    print(f"  FINAL   : {scores['final'] * 100:6.2f} % -> {verdict(scores['final'])}")

    x, y = a["tokens"], b["tokens"]
    pairs = lcs_alignment(x, y)
    only_a, only_b = changed_words(x, y, pairs)
    print(f"  Longest copied phrase: \"{' '.join(longest_common_substring(x, y))}\"")
    print(f"  Words only in A: {' '.join(only_a[:10]) or '-'}")
    print(f"  Words only in B: {' '.join(only_b[:10]) or '-'}")

    for s_a, s_b, score in sentence_level_matches(a["sentences"], b["sentences"], tokenize):
        print(f"  Similar sentences [{score:.2f}]:\n    A: {s_a}\n    B: {s_b}")
    print()


def main():
    # 1. Load the documents
    paths = get_paths()
    names = [os.path.basename(p) for p in paths]
    docs = [preprocess(read_file(p)) for p in paths]
    print(f"{len(names)} documents loaded -> {number_of_pairs(len(names))} pairs to compare\n")

    # 2. Score every pair of documents
    cache = {}

    def score_fn(i, j):
        cache[(i, j)] = analyze_pair(docs[i], docs[j])
        return cache[(i, j)]["final"]

    S = build_similarity_matrix(names, score_fn)

    # 3. Report only the pairs above the threshold
    for (i, j), sc in cache.items():
        if sc["final"] >= THRESHOLD:
            report(names[i], names[j], docs[i], docs[j], sc)

    # 4. Matrix and clusters
    print("SIMILARITY MATRIX")
    print_matrix(names, S)

    graph = build_graph(names, S, THRESHOLD)
    print("\nVertex degrees:", vertex_degrees(graph))
    print("Plagiarism clusters:")
    for c in connected_components(graph):
        print(" ", c, "<-- suspicious group" if len(c) > 1 else "")

    # 5. Charts
    generate_all_plots(names, docs, S, graph, THRESHOLD, cache, out_dir=OUT_DIR)
    print("\nCharts saved in:", OUT_DIR)
    if SHOW_CHARTS:
        show_plots(5)
    else:
        close_plots()


if __name__ == "__main__":
    main()