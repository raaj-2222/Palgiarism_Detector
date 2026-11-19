"""
MAIN DRIVER - integrates all four modules (run this file).

Usage:
    python main.py                               # compares every .txt in sample_docs/
    python main.py --folder my_docs --threshold 0.5
    python main.py --files a.txt b.txt           # compare just two files
    python main.py --delay 5                     # charts pop up after 5 s (default)
    python main.py --no-show                     # only save the charts
"""
import argparse
import os
import sys

from preprocessing import read_file, preprocess, tokenize          # Member 1
from set_similarity import jaccard, dice, overlap, cosine_multiset, set_operations  # Member 2
from lcs_module import (lcs_similarity, lcs_alignment,              # Member 3
                        longest_common_substring, sentence_level_matches,
                        changed_words, lcs_set_analysis, verify_lcs_as_relation)
from graph_cluster import (build_similarity_matrix, build_graph,    # Member 4
                           vertex_degrees, connected_components,
                           relation_properties, print_matrix,
                           number_of_pairs, verdict, graph_summary,
                           generate_all_plots, show_plots, close_plots)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # works from any terminal folder

# Weights for the final combined score (must sum to 1)
WEIGHTS = {"jaccard": 0.20, "ngram": 0.25, "cosine": 0.15, "lcs": 0.40}

SAMPLE_DOCS = {
    "doc1.txt": "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning and self correction. Modern systems use large amounts of data to improve their performance over time. Many industries such as healthcare and finance now depend on these tools.",
    "doc2.txt": "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning and self correction. Modern systems use huge amounts of data to improve their performance with time. Many industries like healthcare and banking now depend on these tools.",
    "doc3.txt": "Machines that imitate human thinking are called artificial intelligence. Such systems learn from large amounts of data to improve their performance over time. Schools also use computers for teaching students.",
    "doc4.txt": "The monsoon season in Mumbai brings heavy rainfall from June to September. Local trains often get delayed and streets fill with water. Street vendors sell hot tea and corn to people waiting for the rain to stop.",
    "doc5.txt": "Machines that imitate human thinking are called artificial intelligence. It includes learning, reasoning and self correction. Such systems learn from large amounts of data to improve their performance over time. Many industries such as healthcare and finance now depend on these tools.",
    "doc6.txt": "Heavy rainfall arrives in Mumbai during the monsoon season from June to September. Local trains often get delayed and streets fill with water. Vendors on the street sell hot tea and corn to people waiting for the rain to stop.",
}


def find_documents(folder):
    """All .txt files in a folder (empty list if the folder does not exist)."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(".txt"))


def ensure_sample_docs(folder):
    """Create the demo documents (never overwrites an existing file)."""
    os.makedirs(folder, exist_ok=True)
    for name, text in SAMPLE_DOCS.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")


def analyze_pair(pa, pb):
    """Compute every metric for two preprocessed documents."""
    scores = {
        "jaccard": jaccard(pa["keywords"], pb["keywords"]),
        "dice": dice(pa["keywords"], pb["keywords"]),
        "overlap": overlap(pa["keywords"], pb["keywords"]),
        "cosine": cosine_multiset(pa["keywords"], pb["keywords"]),
        "ngram": jaccard(pa["ngrams"], pb["ngrams"]),
        "lcs": lcs_similarity(pa["tokens"], pb["tokens"]),
    }
    scores["final"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return scores


def _preview(words, limit=10):
    shown = " ".join(words[:limit])
    return shown + (" ..." if len(words) > limit else "") if words else "-"


def detailed_report(name_a, name_b, pa, pb, scores):
    print("=" * 70)
    print(f"  {name_a}  vs  {name_b}")
    print("=" * 70)
    for k in ("jaccard", "dice", "overlap", "cosine", "ngram", "lcs"):
        print(f"  {k:<10}: {scores[k]*100:6.2f} %")
    print(f"  {'FINAL':<10}: {scores['final']*100:6.2f} %  ->  {verdict(scores['final'])}")

    # Member 2 - set theory on keywords
    ops = set_operations(pa["keywords"], pb["keywords"])
    print(f"\n  |A|={len(set(pa['keywords']))}  |B|={len(set(pb['keywords']))}  "
          f"|A∩B|={len(ops['intersection'])}  |A∪B|={len(ops['union'])}")

    # Member 3 - LCS + set theory on matched positions
    x, y = pa["tokens"], pb["tokens"]
    pairs = lcs_alignment(x, y)
    sub = longest_common_substring(x, y)
    sa = lcs_set_analysis(x, y, pairs)
    rel = verify_lcs_as_relation(x, y, pairs)
    only_a, only_b = changed_words(x, y, pairs)
    print(f"  LCS length (words)            : {len(pairs)}")
    print(f"  Longest exact copied phrase   : \"{' '.join(sub)}\" ({len(sub)} words)")
    print(f"  Matched positions (|M|)       : {sa['matched_in_A']}/{sa['len_A']} in A, "
          f"{sa['matched_in_B']}/{sa['len_B']} in B")
    print(f"  Partition |M|+|U| = |X|       : {sa['partition_ok']}")
    print(f"  LCS is 1-1, order-preserving ⊆ X×Y : {rel['all_hold']}")
    print(f"  Unmatched words in A (X - M)  : {_preview(only_a)}")
    print(f"  Unmatched words in B (Y - M)  : {_preview(only_b)}")

    matches = sentence_level_matches(pa["sentences"], pb["sentences"], tokenize, 0.7)
    if matches:
        print("\n  Suspicious sentence pairs (LCS >= 0.70):")
        for s_a, s_b, sc in matches:
            print(f"    [{sc:.2f}] A: {s_a}")
            print(f"           B: {s_b}")
    print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Plagiarism & Document Similarity Detector")
    ap.add_argument("--folder", default=None, help="folder with .txt files (default: sample_docs next to main.py)")
    ap.add_argument("--files", nargs=2, metavar=("DOC1", "DOC2"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="folder for the chart PNGs (default: output next to main.py)")
    ap.add_argument("--delay", type=int, default=5, help="seconds before the chart windows pop up (default 5)")
    ap.add_argument("--no-show", action="store_true", help="save the charts but do not open windows")
    ap.add_argument("--no-plots", action="store_true", help="skip chart generation")
    args = ap.parse_args()

    if args.files:
        missing = [p for p in args.files if not os.path.isfile(p)]
        if missing:
            print("File not found:", ", ".join(os.path.abspath(p) for p in missing))
            return
        paths = args.files
    else:
        folder = args.folder or os.path.join(BASE_DIR, "sample_docs")
        paths = find_documents(folder)
        if len(paths) < 2 and args.folder is None:
            ensure_sample_docs(folder)
            print(f"Sample documents were missing - created them in:\n  {folder}\n")
            paths = find_documents(folder)
        if len(paths) < 2:
            print(f"Need at least 2 .txt documents, but found {len(paths)} in:\n  {os.path.abspath(folder)}")
            print("Put 2 or more .txt files in that folder, or run:  python main.py --files a.txt b.txt")
            return
        print(f"Reading documents from: {os.path.abspath(folder)}")

    out_dir = args.out or os.path.join(BASE_DIR, "output")
    names = [os.path.basename(p) for p in paths]
    docs = [preprocess(read_file(p)) for p in paths]

    print(f"Documents loaded: {len(names)}  ->  {number_of_pairs(len(names))} pairs to compare\n")

    cache = {}
    def score_fn(i, j):
        cache[(i, j)] = analyze_pair(docs[i], docs[j])
        return cache[(i, j)]["final"]

    S = build_similarity_matrix(names, score_fn)

    # ---- Detailed pair reports (only pairs above threshold) ----
    for (i, j), sc in cache.items():
        if sc["final"] >= args.threshold or len(names) == 2:
            detailed_report(names[i], names[j], docs[i], docs[j], sc)

    # ---- Matrix ----
    print("SIMILARITY MATRIX (final combined score)")
    print_matrix(names, S)

    # ---- Graph analysis ----
    graph = build_graph(names, S, args.threshold)
    print(f"\nSIMILARITY GRAPH (edge if score >= {args.threshold})")
    for v, nbrs in graph.items():
        print(f"  {v}: {nbrs if nbrs else '-'}")
    print("\nVertex degrees :", vertex_degrees(graph))

    print("Plagiarism clusters (connected components):")
    for c in connected_components(graph):
        tag = "  <-- suspicious group" if len(c) > 1 else ""
        print(f"  {c}{tag}")

    print("\nGraph summary:")
    for k, v in graph_summary(graph).items():
        print(f"  {k}: {v}")

    print("\nRelation properties of 'is similar to':", relation_properties(S, args.threshold))

    # ---- Charts ----
    if not args.no_plots:
        saved = generate_all_plots(names, docs, S, graph, args.threshold, cache, out_dir=out_dir)
        print(f"\nCharts saved in: {out_dir}")
        for path in saved:
            print("  ", os.path.basename(path))
        if args.no_show:
            close_plots()
        else:
            print()
            show_plots(args.delay)


if __name__ == "__main__":
    main()
