"""
MEMBER 3 - LCS & SET THEORY ON SEQUENCES MODULE
-----------------------------------------------
Responsibility : Compare the ORDER of words in two documents. Finds the
                 Longest Common Subsequence (LCS) and then analyses the result
                 with core set theory (same style as set_similarity.py).

Discrete Maths used:
  * Subsequence vs substring
  * Recurrence relation + dynamic programming + backtracking
        L[i][j] = L[i-1][j-1] + 1               if X[i] == Y[j]
                = max(L[i-1][j], L[i][j-1])     otherwise
  * Sets of positions: matched set M, unmatched set U = X - M (set difference)
  * Partition / addition principle:  |M| + |U| = |X|,  M ∪ U = X,  M ∩ U = ∅
  * Cartesian product X × Y (product rule |X×Y| = m·n) - the LCS is a subset of it
  * Functions: the LCS matching is one-to-one (injective) and order preserving
"""


# =====================================================================
#  CORE LCS (dynamic programming)
# =====================================================================
def lcs_table(x, y):
    """Build the DP table of LCS lengths for sequences x and y."""
    m, n = len(x), len(y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L


def lcs_alignment(x, y):
    """Backtrack the table and return the index pairs (i, j) of matched words."""
    L = lcs_table(x, y)
    i, j = len(x), len(y)
    pairs = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def lcs_sequence(x, y):
    """Recover the actual LCS (the words themselves)."""
    return [x[i] for i, _ in lcs_alignment(x, y)]


def lcs_similarity(x, y):
    """Normalised LCS score in [0,1]:   2 * |LCS| / (|X| + |Y|)"""
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return 2 * len(lcs_sequence(x, y)) / (len(x) + len(y))


def longest_common_substring(x, y):
    """Longest CONTIGUOUS run of common words (strong evidence of copy-paste)."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    best_len, end_i = 0, 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len, end_i = curr[j], i
        prev = curr
    return x[end_i - best_len:end_i]


def sentence_level_matches(sents_a, sents_b, tokenizer, threshold=0.7):
    """Best LCS match in B for every sentence of A; keep pairs >= threshold."""
    matches = []
    for sa in sents_a:
        ta = tokenizer(sa)
        best_score, best_sb = 0.0, None
        for sb in sents_b:
            score = lcs_similarity(ta, tokenizer(sb))
            if score > best_score:
                best_score, best_sb = score, sb
        if best_sb is not None and best_score >= threshold:
            matches.append((sa, best_sb, round(best_score, 3)))
    return matches


# =====================================================================
#  CORE DISCRETE MATHS (SET THEORY) APPLIED TO THE LCS
#  Pass `pairs=lcs_alignment(x, y)` to avoid recomputing the DP table.
# =====================================================================
def _pairs(x, y, pairs):
    return lcs_alignment(x, y) if pairs is None else pairs


def matched_positions(x, y, pairs=None):
    """Sets of matched positions  Mx ⊆ {0..m-1}  and  My ⊆ {0..n-1}."""
    pairs = _pairs(x, y, pairs)
    return {i for i, _ in pairs}, {j for _, j in pairs}


def unmatched_positions(x, y, pairs=None):
    """
    Complement by SET DIFFERENCE:  Ux = X - Mx ,  Uy = Y - My.
    These are the words that were inserted, deleted or changed.
    """
    mx, my = matched_positions(x, y, pairs)
    return set(range(len(x))) - mx, set(range(len(y))) - my


def changed_words(x, y, pairs=None):
    """The actual words at the unmatched positions (in document order)."""
    ux, uy = unmatched_positions(x, y, pairs)
    return [x[i] for i in sorted(ux)], [y[j] for j in sorted(uy)]


def lcs_set_analysis(x, y, pairs=None):
    """
    Cardinalities of the matched / unmatched sets and a check of the
    partition (addition) principle:
        Mx ∪ Ux = X ,  Mx ∩ Ux = ∅ ,  |Mx| + |Ux| = |X|   (same for Y)
    """
    pairs = _pairs(x, y, pairs)
    mx, my = matched_positions(x, y, pairs)
    ux, uy = unmatched_positions(x, y, pairs)
    ux_all, uy_all = set(range(len(x))), set(range(len(y)))
    partition_ok = (
        (mx | ux == ux_all) and not (mx & ux) and len(mx) + len(ux) == len(x)
        and (my | uy == uy_all) and not (my & uy) and len(my) + len(uy) == len(y)
    )
    return {
        "len_A": len(x), "len_B": len(y),
        "matched_in_A": len(mx), "matched_in_B": len(my),
        "unmatched_in_A": len(ux), "unmatched_in_B": len(uy),
        "coverage_A": len(mx) / len(x) if x else 1.0,
        "coverage_B": len(my) / len(y) if y else 1.0,
        "partition_ok": partition_ok,
    }


def verify_lcs_as_relation(x, y, pairs=None):
    """
    The LCS matching is a relation  R ⊆ X × Y  (|X × Y| = m·n by the product rule).
    It is also a function that is one-to-one (injective) and order preserving.
    """
    pairs = _pairs(x, y, pairs)
    in_product = all(0 <= i < len(x) and 0 <= j < len(y) for i, j in pairs)
    one_to_one = len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
    order_preserving = all(pairs[k][0] < pairs[k + 1][0] and pairs[k][1] < pairs[k + 1][1]
                           for k in range(len(pairs) - 1))
    return {
        "product_size |X x Y|": len(x) * len(y),
        "subset_of_product": in_product,
        "one_to_one": one_to_one,
        "order_preserving": order_preserving,
        "all_hold": in_product and one_to_one and order_preserving,
    }


if __name__ == "__main__":
    X = "the cat sat on the mat".split()
    Y = "the black cat sat quietly on a mat".split()
    pairs = lcs_alignment(X, Y)
    print("LCS                :", lcs_sequence(X, Y))
    print("LCS score          :", round(lcs_similarity(X, Y), 3))
    print("Longest substring  :", longest_common_substring(X, Y))
    print("DP table:")
    for row in lcs_table(X, Y):
        print("  ", row)
    print("\n-- Set theory on the LCS --")
    print("Matched positions  :", matched_positions(X, Y, pairs))
    print("Unmatched (X - M)  :", unmatched_positions(X, Y, pairs))
    print("Changed words      :", changed_words(X, Y, pairs))
    print("Set analysis       :", lcs_set_analysis(X, Y, pairs))
    print("Relation check     :", verify_lcs_as_relation(X, Y, pairs))
