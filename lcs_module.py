"""LCS module - compares the ORDER of words in two documents."""


def lcs_table(x, y):
    """Table where L[i][j] = LCS length of x[:i] and y[:j]."""
    L = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L


def lcs_alignment(x, y):
    """Walk back through the table, return matched positions (i, j)."""
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


def lcs_similarity(x, y):
    """Score from 0 to 1: 2 * LCS length / (len(x) + len(y))"""
    if not x or not y:
        return 1.0 if not x and not y else 0.0
    return 2 * len(lcs_alignment(x, y)) / (len(x) + len(y))


def longest_common_substring(x, y):
    """Longest run of words that appear together in both (copy-paste check)."""
    best_len, end = 0, 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        curr = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len, end = curr[j], i
        prev = curr
    return x[end - best_len:end]


def changed_words(x, y, pairs):
    """Words NOT matched by the LCS (set difference: all - matched)."""
    matched_x = {i for i, _ in pairs}
    matched_y = {j for _, j in pairs}
    return ([w for i, w in enumerate(x) if i not in matched_x],
            [w for j, w in enumerate(y) if j not in matched_y])


def sentence_level_matches(sents_a, sents_b, tokenizer, threshold=0.7):
    """For each sentence in A, find the best match in B (if score >= threshold)."""
    matches = []
    for sa in sents_a:
        best_score, best_sb = 0, None
        for sb in sents_b:
            score = lcs_similarity(tokenizer(sa), tokenizer(sb))
            if score > best_score:
                best_score, best_sb = score, sb
        if best_score >= threshold:
            matches.append((sa, best_sb, round(best_score, 3)))
    return matches