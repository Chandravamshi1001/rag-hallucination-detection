"""Verify each claim against up to three TF-IDF evidence chunks with CPU NLI."""

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from time import monotonic

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tfidf_retrieval_baseline import (
    CLAIM_FIELDS, RESULTS_DIR, SUPPORTED, TASKS, UNSUPPORTED,
    classification_metrics, fit_vectorizer, labels_and_scores, load_split,
    source_similarities, validation_thresholds, write_csv,
)


MODEL_ID = "Sayan01/tiny-bert-mnli-distilled"
TOP_K = 3
NLI_LABELS = ("entailment", "contradiction", "neutral")
PREDICTION_FIELDS = (
    *CLAIM_FIELDS, "true_label", "predicted_label", "support_score", "contradiction_score",
    "best_evidence_chunk_index", "best_evidence_chunk", "retrieved_chunk_count",
)
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass(frozen=True)
class PairScores:
    entailment: float
    contradiction: float
    neutral: float


def model_label_ids(model):
    labels = {str(label).strip().lower(): int(index)
              for index, label in model.config.id2label.items()}
    if set(labels) != set(NLI_LABELS) or set(labels.values()) != set(range(len(NLI_LABELS))):
        raise ValueError(f"Expected the three named NLI labels, got {model.config.id2label!r}")
    return [labels[label] for label in NLI_LABELS]


def retrieve_pairs(vectorizer, claims, chunks_by_source):
    pairs = defaultdict(list)
    retrieved = [[] for _ in claims]
    for chunks, indices, similarities in source_similarities(vectorizer, claims, chunks_by_source):
        for index, scores in zip(indices, similarities):
            # Stable sorting resolves similarity ties by the original chunk index.
            for chunk_index in np.argsort(-scores, kind="stable")[:TOP_K]:
                chunk_index = int(chunk_index)
                pair = (chunks[chunk_index], claims[index]["claim_text"])
                pairs[pair].append((index, len(retrieved[index])))
                retrieved[index].append((chunk_index, pair[0], None))
    return pairs, retrieved


@torch.inference_mode()
def verify_split(split, vectorizer, claims, chunks, model, tokenizer, label_ids, batch_size):
    pair_uses, retrieved = retrieve_pairs(vectorizer, claims, chunks)
    pairs = list(pair_uses)
    print(f"{split}: {len(claims):,} claims, {sum(map(len, retrieved)):,} retrieved pairs "
          f"({len(pairs):,} unique).", flush=True)
    started = monotonic()
    window = batch_size * 32
    max_length = min(512, tokenizer.model_max_length)
    for offset in range(0, len(pairs), window):
        block = pairs[offset:offset + window]
        premises, hypotheses = zip(*block)
        encoded = tokenizer(list(premises), list(hypotheses),
                            truncation="longest_first", max_length=max_length)
        # Group similar lengths inside each window to spend less time on padding.
        order = sorted(range(len(block)), key=lambda index: len(encoded["input_ids"][index]))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            features = [{key: values[index] for key, values in encoded.items()} for index in indices]
            inputs = tokenizer.pad(features, padding=True, return_tensors="pt")
            probabilities = model(**inputs).logits.softmax(dim=-1)[:, label_ids].tolist()
            for index, scores in zip(indices, probabilities):
                if not np.isfinite(scores).all():
                    raise ValueError(f"Nonfinite NLI probabilities in {split}")
                result = PairScores(*scores)
                for claim_index, rank in pair_uses[block[index]]:
                    chunk_index, text, _ = retrieved[claim_index][rank]
                    retrieved[claim_index][rank] = (chunk_index, text, result)
        done = offset + len(block)
        print(f"{split}: {done:,}/{len(pairs):,} unique pairs scored "
              f"in {monotonic() - started:.0f}s.", flush=True)

    for claim, evidence in zip(claims, retrieved):
        best_index, best_text, best_scores = max(evidence, key=lambda item: item[2].entailment)
        claim.update(
            support_score=best_scores.entailment,
            contradiction_score=max(scores.contradiction for _, _, scores in evidence),
            best_evidence_chunk_index=best_index,
            best_evidence_chunk=best_text,
            retrieved_chunk_count=len(evidence),
        )
    return retrieved


def evaluate(claims, score_field):
    labels, scores = labels_and_scores(claims, score_field)
    predictions = [claim["predicted_label"] == UNSUPPORTED for claim in claims]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    return {
        **classification_metrics(tn, fp, fn, tp),
        "roc_auc": roc_auc_score(labels, -scores) if labels.any() and not labels.all() else None,
    }


def read_tfidf_predictions(test):
    path = RESULTS_DIR / "tfidf_test_predictions.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    baseline = {row["claim_id"]: row for row in rows}
    if len(baseline) != len(rows) or len(rows) != len(test):
        raise ValueError("TF-IDF predictions must cover the same unique test claims")
    for claim in test:
        row = baseline.get(claim["claim_id"])
        if row is None or any(row[key] != str(claim[key]) for key in (*CLAIM_FIELDS, "true_label")):
            raise ValueError(f"TF-IDF test prediction mismatch for {claim['claim_id']}")
    return rows


def format_metric(value):
    return f"{value:.6f}" if value is not None else "N/A"


def report_metrics(test, baseline, best, validation_count, model, args):
    lines = [
        "RAGTruth NLI evidence-verification baseline",
        f"Model: {args.model} on CPU ({args.threads} threads, batch size {args.batch_size}).",
        f"Label mapping: {model.config.id2label}",
        "TF-IDF fits training context only and retrieves up to three chunks per claim "
        "from its own source. NLI receives only each chunk and claim as input.",
        "Neutral means that the evidence does not establish the claim. "
        "Neutral and contradiction both count as unsupported in the binary task.",
        "Support is the maximum entailment probability. Contradiction is reported separately.",
        "Prediction is unsupported when support is strictly below the frozen cutoff.",
        f"Best validation threshold: {best['threshold']!r}",
        f"Best validation F1: {best['f1']:.6f} ({validation_count:,} claims).",
        "Validation ties use the smallest threshold. ROC-AUC uses negative support for NLI "
        "and negative similarity for TF-IDF.",
        "Confusion matrices use true rows, predicted columns, and supported first.",
    ]
    for scope in ("Overall", *TASKS):
        nli_rows = test if scope == "Overall" else [row for row in test if row["task_type"] == scope]
        tfidf_rows = baseline if scope == "Overall" else [
            row for row in baseline if row["task_type"] == scope
        ]
        lines.extend([
            "", f"{scope} ({len(nli_rows):,} claims)",
            f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'ROC-AUC':>10}  Confusion matrix",
        ])
        for name, rows, score_field in (
            ("TF-IDF", tfidf_rows, "max_similarity"), ("NLI", nli_rows, "support_score")
        ):
            metrics = evaluate(rows, score_field)
            values = " ".join(f"{format_metric(metrics[key]):>10}" for key in METRICS)
            matrix = f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]"
            lines.append(f"{name:<10} {values}  {matrix}")
    return "\n".join(lines) + "\n"


def error_report(test, retrieved):
    sections = []
    for title, label in (("False positives", SUPPORTED), ("False negatives", UNSUPPORTED)):
        errors = [(claim, evidence) for claim, evidence in zip(test, retrieved)
                  if claim["true_label"] == label and claim["predicted_label"] != label]
        sections.append(f"{title} (showing {min(5, len(errors))} of {len(errors):,})")
        for claim, evidence in errors[:5]:
            pair_rows = [
                f"Chunk {index}: entailment {scores.entailment:.6f}, "
                f"contradiction {scores.contradiction:.6f}, neutral {scores.neutral:.6f}"
                for index, _, scores in evidence
            ]
            sections.append("\n".join([
                f"{claim['claim_id']} ({claim['task_type']}, source {claim['source_id']})",
                f"Claim: {claim['claim_text']}",
                f"Best evidence (chunk {claim['best_evidence_chunk_index']}): "
                f"{claim['best_evidence_chunk']}",
                *pair_rows,
            ]))
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model identifier")
    parser.add_argument("--model-path", help="Local snapshot of the selected checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and threads must be positive")

    torch.set_num_threads(args.threads)
    print(f"Loading NLI checkpoint {args.model} on CPU.", flush=True)
    checkpoint = args.model_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).cpu().eval()
    label_ids = model_label_ids(model)

    print("Fitting TF-IDF on training context chunks.", flush=True)
    seen_sources = set()
    train, train_chunks = load_split("train", seen_sources)
    vectorizer = fit_vectorizer(train_chunks)
    del train, train_chunks

    validation, validation_chunks = load_split("validation", seen_sources)
    verify_split("validation", vectorizer, validation, validation_chunks,
                 model, tokenizer, label_ids, args.batch_size)
    thresholds = validation_thresholds(validation, "support_score")
    best = max(thresholds, key=lambda row: row["f1"])
    threshold = best["threshold"]
    print(f"Frozen validation threshold: {threshold!r} (F1 {best['f1']:.6f}).", flush=True)

    test, test_chunks = load_split("test", seen_sources)
    baseline = read_tfidf_predictions(test)
    retrieved = verify_split("test", vectorizer, test, test_chunks,
                             model, tokenizer, label_ids, args.batch_size)
    for claim in test:
        claim["predicted_label"] = UNSUPPORTED if claim["support_score"] < threshold else SUPPORTED

    report = report_metrics(test, baseline, best, len(validation), model, args)
    write_csv(RESULTS_DIR / "nli_validation_thresholds.csv", thresholds, list(best))
    write_csv(RESULTS_DIR / "nli_test_predictions.csv",
              ({key: claim[key] for key in PREDICTION_FIELDS} for claim in test), PREDICTION_FIELDS)
    (RESULTS_DIR / "nli_baseline_metrics.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "nli_error_examples.txt").write_text(error_report(test, retrieved), encoding="utf-8")
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
