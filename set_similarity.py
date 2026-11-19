"""
MEMBER 2 - SET THEORY & SIMILARITY MEASURES MODULE
--------------------------------------------------
Responsibility : Measure how similar two documents are using set theory
                 and multisets.

Discrete Maths used:
  * Union, Intersection, Difference, Symmetric Difference
  * Cardinality |A|
  * Jaccard index  J(A,B) = |A ∩ B| / |A ∪ B|
  * Dice coefficient, Overlap coefficient
  * Multisets (bags) + vectors -> Cosine similarity
  * Inclusion-Exclusion:  |A ∪ B| = |A| + |B| - |A ∩ B|
"""
import math
from collections import Counter


def set_operations(a, b):
    """Return all basic set operations on two sets."""
    a, b = set(a), set(b)
    return {
        "union": a | b,
        "intersection": a & b,
        "only_in_A (A - B)": a - b,
        "only_in_B (B - A)": b - a,
        "symmetric_difference": a ^ b,
    }


def jaccard(a, b):
    """J(A,B) = |A ∩ B| / |A ∪ B|   (0 = totally different, 1 = identical)."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_distance(a, b):
    """d(A,B) = 1 - J(A,B). This is a valid metric on sets."""
    return 1 - jaccard(a, b)


def dice(a, b):
    """Dice = 2|A ∩ B| / (|A| + |B|)."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def overlap(a, b):
    """Overlap = |A ∩ B| / min(|A|, |B|). Good at catching a copied small part."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cosine_multiset(tokens_a, tokens_b):
    """
    Treat each document as a multiset (bag of words) = a vector of counts.
    cos(θ) = (u · v) / (|u| |v|)
    """
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    vocab = set(ca) | set(cb)
    dot = sum(ca[w] * cb[w] for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def verify_inclusion_exclusion(a, b):
    """Check the identity |A ∪ B| = |A| + |B| - |A ∩ B| (nice for demo/viva)."""
    a, b = set(a), set(b)
    return len(a | b) == len(a) + len(b) - len(a & b)


if __name__ == "__main__":
    A = "the quick brown fox jumps".split()
    B = "the quick red fox sleeps".split()
    print("Set operations:", set_operations(A, B))
    print("Jaccard :", round(jaccard(A, B), 3))
    print("Dice    :", round(dice(A, B), 3))
    print("Overlap :", round(overlap(A, B), 3))
    print("Cosine  :", round(cosine_multiset(A, B), 3))
    print("Inclusion-Exclusion holds:", verify_inclusion_exclusion(A, B))
