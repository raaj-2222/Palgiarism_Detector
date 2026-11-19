"""
MEMBER 1 - TEXT PREPROCESSING MODULE
------------------------------------
Responsibility : Convert raw documents into mathematical objects
                 (sequences, sets, n-gram "shingles") that the other modules use.

Discrete Maths used:
  * Sequences / strings  -> a document is an ordered sequence of words
  * Sets                 -> unique keywords of a document
  * n-grams (shingles)   -> overlapping subsequences of length n
"""
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "of", "to",
    "in", "on", "at", "for", "with", "and", "or", "but", "if", "it", "its", "this",
    "that", "these", "those", "as", "by", "from", "so", "than", "then", "there",
    "which", "who", "whom", "do", "does", "did", "has", "have", "had", "can", "will",
}


def read_file(path):
    """Read a text file and return its content as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_sentences(text):
    """Split raw text into sentences using . ! ? as delimiters."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_text(text):
    """Lower-case and remove punctuation / extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, remove_stopwords=False):
    """Return the document as an ORDERED list (sequence) of words."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def generate_ngrams(tokens, n=3):
    """
    Build the list of n-grams (shingles).
    Example: ['a','b','c','d'], n=2 -> [('a','b'), ('b','c'), ('c','d')]
    A document with k tokens has k - n + 1 n-grams.
    """
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def preprocess(text, n=3):
    """Run the full pipeline and return everything the other modules need."""
    tokens = tokenize(text)                              # ordered sequence (for LCS)
    keywords = tokenize(text, remove_stopwords=True)     # for sets / multisets
    return {
        "raw": text,
        "sentences": split_sentences(text),
        "tokens": tokens,
        "keywords": keywords,
        "ngrams": generate_ngrams(tokens, n),
    }


if __name__ == "__main__":
    sample = "Discrete maths is fun. Sets, graphs and logic are everywhere!"
    info = preprocess(sample, n=2)
    for k, v in info.items():
        print(f"{k:10}: {v}")
