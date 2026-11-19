# Plagiarism & Document Similarity Detector (Discrete Mathematics Mini Project)

Install:  `pip install matplotlib`   (numpy comes with it)

Run:
  python main.py                      -> report, saves 4 charts, charts pop up after 5 s
  python main.py --delay 3            -> pop up after 3 s
  python main.py --no-show            -> only save the PNGs (output/)
  python main.py --files a.txt b.txt  -> compare two documents
  python main.py --folder my_docs --threshold 0.4

If sample_docs/ is missing or empty, main.py creates 6 demo documents automatically.

| Member | File | Discrete Maths concepts |
|---|---|---|
| 1 | preprocessing.py | Sequences, sets, n-grams (shingles) |
| 2 | set_similarity.py | Set operations, Jaccard, Dice, Overlap, multisets, cosine, inclusion-exclusion |
| 3 | lcs_module.py | LCS (recurrence + DP), set difference/complement of matched positions, partition principle, Cartesian product X x Y, injective + order-preserving function |
| 4 | graph_cluster.py | Graphs, BFS, components, handshaking lemma, relations, C(n,2) + 4 matplotlib charts |
| All | main.py | Integration (weighted score + report) |

Final score = 0.20*Jaccard + 0.25*3-gram Jaccard + 0.15*Cosine + 0.40*LCS

Charts: 01 similarity graph | 02 metric comparison | 03 common vs unique words | 04 Venn diagram
