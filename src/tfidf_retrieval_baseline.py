"""Retrieve evidence for RAGTruth claims and evaluate a similarity cutoff."""

import csv
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score

from explore_context import PROJECT_ROOT


CLAIMS_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CHUNK_SIZE = 120
OVERLAP = 30
TASKS = ("QA", "Summary", "Data2txt")
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CLAIM_FIELDS = ("response_id", "source_id", "claim_id", "task_type", "claim_text")
PREDICTION_FIELDS = (
    *CLAIM_FIELDS, "true_label", "predicted_label", "max_similarity",
    "best_chunk_index", "best_evidence_chunk",
)


def chunk_context(source_info):
    text = source_info if isinstance(source_info, str) else json.dumps(
        source_info, ensure_ascii=False
    )
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), CHUNK_SIZE - OVERLAP):
        end = min(start + CHUNK_SIZE, len(words))
        # Slice the original text so whitespace and formatting inside chunks survive.
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break


def load_split(split, seen_sources):
    claims = []
    chunks_by_source = {}
    path = CLAIMS_DIR / f"{split}_claims.jsonl"
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                claim = {key: row[key] for key in CLAIM_FIELDS}
                source_id = claim["source_id"]
                label = row["claim_label"]
                if label not in (SUPPORTED, UNSUPPORTED):
                    raise ValueError(f"Unexpected label {label!r}")
                if row["split"] != split or claim["task_type"] not in TASKS:
                    raise ValueError("Unexpected split or task type")
                if not isinstance(claim["claim_text"], str) or not claim["claim_text"].strip():
                    raise ValueError("Expected nonempty claim text")
                if source_id is None or "source_info" not in row:
                    raise ValueError("Missing source ID or context")
                if source_id not in chunks_by_source:
                    if source_id in seen_sources:
                        raise ValueError(f"Source {source_id!r} occurs in multiple splits")
                    chunks = list(chunk_context(row["source_info"]))
                    if not chunks:
                        raise ValueError(f"Source {source_id!r} has no text chunks")
                    chunks_by_source[source_id] = chunks
                    seen_sources.add(source_id)
                claim["true_label"] = label
                claims.append(claim)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not claims:
        raise ValueError(f"No claims in {path}")
    return claims, chunks_by_source


def fit_vectorizer(chunks_by_source):
    vectorizer = TfidfVectorizer(
        lowercase=False, ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+(?:[’']\w+)*\b",
        stop_words=None, norm="l2",
    )
    vectorizer.fit(chunk for chunks in chunks_by_source.values() for chunk in chunks)
    return vectorizer


def source_similarities(vectorizer, claims, chunks_by_source):
    claim_vectors = vectorizer.transform(claim["claim_text"] for claim in claims)
    indices_by_source = defaultdict(list)
    for index, claim in enumerate(claims):
        indices_by_source[claim["source_id"]].append(index)

    for source_id, indices in indices_by_source.items():
        chunks = chunks_by_source[source_id]
        chunk_vectors = vectorizer.transform(chunks)
        # L2-normalized TF-IDF makes the dot product equal to cosine similarity.
        similarities = (claim_vectors[indices] @ chunk_vectors.T).toarray()
        yield chunks, indices, similarities


def score_claims(vectorizer, claims, chunks_by_source):
    for chunks, indices, similarities in source_similarities(vectorizer, claims, chunks_by_source):
        for index, scores in zip(indices, similarities):
            best = int(scores.argmax())
            claims[index].update(
                max_similarity=float(scores[best]),
                best_chunk_index=best,
                best_evidence_chunk=chunks[best],
            )


def labels_and_scores(claims, score_field="max_similarity"):
    return (
        np.array([claim["true_label"] == UNSUPPORTED for claim in claims], dtype=bool),
        np.array([claim[score_field] for claim in claims], dtype=float),
    )


def classification_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    return {
        "accuracy": (tn + tp) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def validation_thresholds(claims, score_field="max_similarity"):
    labels, scores = labels_and_scores(claims, score_field)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Validation threshold selection requires both labels")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    cumulative_tp = np.cumsum(labels[order])
    # A cutoff just above each distinct score adds the entire tied score group.
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    rows = [{"threshold": float(sorted_scores[0]),
             **classification_metrics(negatives, 0, positives, 0)}]
    for end in group_ends:
        tp = int(cumulative_tp[end])
        fp = int(end) + 1 - tp
        threshold = float(np.nextafter(sorted_scores[end], np.inf))
        rows.append({"threshold": threshold,
                     **classification_metrics(negatives - fp, fp, positives - tp, tp)})
    return rows


def metrics_section(name, claims, threshold):
    if not claims:
        return f"{name}\nClaims: 0\nMetrics unavailable\n"
    labels, scores = labels_and_scores(claims)
    tn, fp, fn, tp = confusion_matrix(labels, scores < threshold, labels=[False, True]).ravel()
    metrics = classification_metrics(tn, fp, fn, tp)
    # Lower similarity is stronger evidence for the positive class.
    auc = f"{roc_auc_score(labels, -scores):.6f}" if labels.any() and not labels.all() else "N/A"
    return "\n".join([
        name,
        f"Claims: {len(claims):,} ({tp + fn:,} unsupported)",
        *(f"{key.capitalize()}: {metrics[key]:.6f}"
          for key in ("accuracy", "precision", "recall", "f1")),
        f"ROC-AUC (negative similarity): {auc}",
        "Confusion matrix (rows true, columns predicted; supported, unsupported):",
        f"[[{tn}, {fp}], [{fn}, {tp}]]",
        "",
    ])


def error_examples(claims):
    sections = []
    for title, true_label, predicted_label in (
        ("False positives", SUPPORTED, UNSUPPORTED),
        ("False negatives", UNSUPPORTED, SUPPORTED),
    ):
        errors = [
            claim for claim in claims
            if claim["true_label"] == true_label and claim["predicted_label"] == predicted_label
        ]
        sections.append(f"{title} (showing {min(len(errors), 5)} of {len(errors):,})")
        for claim in errors[:5]:
            sections.append(
                f"{claim['claim_id']} ({claim['task_type']}, source {claim['source_id']}, "
                f"similarity {claim['max_similarity']:.6f}, chunk {claim['best_chunk_index']})\n"
                f"Claim: {claim['claim_text']}\n"
                f"Best evidence: {claim['best_evidence_chunk']}"
            )
    return "\n\n".join(sections) + "\n"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    seen_sources = set()
    train, train_chunks = load_split("train", seen_sources)
    vectorizer = fit_vectorizer(train_chunks)
    # Score training claims as well, without using their labels or scores to tune.
    score_claims(vectorizer, train, train_chunks)
    train_count = len(train)
    train_source_count = len(train_chunks)
    train_chunk_count = sum(map(len, train_chunks.values()))
    del train, train_chunks

    validation, validation_chunks = load_split("validation", seen_sources)
    score_claims(vectorizer, validation, validation_chunks)
    thresholds = validation_thresholds(validation)
    # Rows are ascending, so ties retain the smallest cutoff.
    best = max(thresholds, key=lambda row: row["f1"])
    threshold = best["threshold"]

    test, test_chunks = load_split("test", seen_sources)
    score_claims(vectorizer, test, test_chunks)
    for claim in test:
        claim["predicted_label"] = UNSUPPORTED if claim["max_similarity"] < threshold else SUPPORTED

    report = "\n".join([
        "RAGTruth TF-IDF evidence-retrieval baseline",
        f"Training: {train_count:,} claims, {train_source_count:,} sources, "
        f"{train_chunk_count:,} chunks used to fit TF-IDF.",
        f"Context chunks: up to {CHUNK_SIZE} whitespace-delimited words "
        f"with {OVERLAP}-word overlap. Chunk indices are zero-based.",
        "Case-sensitive word unigrams and bigrams retain numbers and negation. "
        "Each claim is compared only with chunks from its own source.",
        "Positive class: unsupported. Predict unsupported when similarity < threshold.",
        f"Best validation threshold: {threshold!r}",
        f"Best validation F1: {best['f1']:.6f} ({len(validation):,} claims).",
        "Validation ties use the smallest cutoff.",
        "",
        metrics_section("Test overall", test, threshold),
        *(metrics_section(f"Test {task}", [row for row in test if row["task_type"] == task],
                          threshold) for task in TASKS),
        "This baseline measures retrieval similarity. High similarity does not prove "
        "factual support. A similar claim can still hallucinate by changing a date, "
        "number, name, negation, or null/unknown value.",
    ]) + "\n"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "tfidf_validation_thresholds.csv", thresholds, list(best))
    write_csv(RESULTS_DIR / "tfidf_test_predictions.csv", test, PREDICTION_FIELDS)
    (RESULTS_DIR / "tfidf_baseline_metrics.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "tfidf_error_examples.txt").write_text(error_examples(test), encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
