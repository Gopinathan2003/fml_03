"""A small multinomial Naive Bayes spam classifier, implemented from scratch.

Run ``python3 classifier.py`` from this directory. It trains on the included
UCI SMS Spam Collection and classifies every email*.txt file in ./test.
"""

from collections import Counter
from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "data" / "SMSSpamCollection"
TEST_DIR = ROOT / "test"
ALPHA = 1.0


def tokenize(text):
    """Lowercase text and keep word-like tokens, including simple contractions."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def read_training_data(path):
    examples = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            label, separator, message = line.partition("\t")
            if not separator or label not in {"ham", "spam"}:
                raise ValueError(f"Bad training row at line {line_number}")
            examples.append((label, tokenize(message)))
    if not examples:
        raise ValueError("The training file has no examples")
    return examples


class MultinomialNaiveBayes:
    """Two-class multinomial NB with Laplace-smoothed token probabilities."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, examples):
        self.document_counts = Counter()
        self.token_counts = {"ham": Counter(), "spam": Counter()}
        self.total_tokens = Counter()
        vocabulary = set()
        for label, tokens in examples:
            self.document_counts[label] += 1
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            vocabulary.update(tokens)
        self.vocabulary = vocabulary
        self.total_documents = sum(self.document_counts.values())
        if not self.total_documents or not vocabulary:
            raise ValueError("Training data must contain documents and tokens")
        return self

    def predict(self, tokens):
        vocabulary_size = len(self.vocabulary)
        scores = {}
        for label in ("ham", "spam"):
            # Add-one smoothing also prevents a zero probability for unseen words.
            prior = self.document_counts[label] / self.total_documents
            score = math.log(prior)
            denominator = self.total_tokens[label] + self.alpha * vocabulary_size
            for token, repetitions in Counter(tokens).items():
                if token not in self.vocabulary:
                    continue
                count = self.token_counts[label][token]
                probability = (count + self.alpha) / denominator
                score += repetitions * math.log(probability)
            scores[label] = score
        return "spam" if scores["spam"] > scores["ham"] else "ham"


def classify_folder(model, folder=TEST_DIR):
    """Print and return predictions for email.txt, email1.txt, email2.txt, etc."""
    paths = [p for p in folder.glob("email*.txt") if p.is_file()]
    paths.sort(key=lambda p: (p.name != "email.txt", p.name))
    if not paths:
        print(f"No email*.txt files found in {folder}")
        return []
    results = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        label = model.predict(tokenize(text))
        assignment_label = 1 if label == "spam" else 0
        results.append((path.name, assignment_label))
        print(f"{path.name}: {assignment_label}")
    return results


def main():
    examples = read_training_data(TRAINING_FILE)
    model = MultinomialNaiveBayes(alpha=ALPHA).fit(examples)
    print(f"Trained on {len(examples)} messages; vocabulary size {len(model.vocabulary)}")
    classify_folder(model)


if __name__ == "__main__":
    main()
