"""Train a CPU logistic regression detector on retrieved claim-evidence features."""

import argparse
import os
import re
from functools import lru_cache

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nli_evidence_baseline import (
    METRICS, MODEL_ID, evaluate, format_metric, model_label_ids,
    read_baseline_predictions, verify_split,
)
from tfidf_retrieval_baseline import (
    CLAIM_FIELDS, RESULTS_DIR, SUPPORTED, TASKS, UNSUPPORTED, fit_vectorizer,
    load_split, validation_thresholds, write_csv,
)


WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)")
NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve".split(),
    map(str, range(13)),
))
MONTHS = ("january february march april may june july august september october "
          "november december").split()
MONTH_IDS = {alias: index for index, month in enumerate(MONTHS, 1)
             for alias in (month, month[:3])}
MONTH_PATTERN = "|".join(sorted(MONTH_IDS, key=len, reverse=True))
MONTH_DAY = re.compile(
    rf"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
    rf"|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})\b", re.I,
)
ISO_DATE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|without|none)\b|n['’]t\b", re.I)
UNKNOWN = re.compile(
    r"\b(?:null|unknown|unclear|unspecified|missing|not (?:available|provided|clear|known)"
    r"|no information)\b|\bn/a\b", re.I,
)
NULL_FIELD = re.compile(r'"([^"\\]+)"\s*:\s*(?:null\b|""|"(?:unknown|n/a|not available)")', re.I)
NAME = re.compile(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b')
ANCHOR = re.compile(r"\d[\d,]*(?:\.\d+)?%?|[^\W\d_]+")
STOP = set("a an the of in on at to and or for from with is are was were be been "
           "by as that this it its he she they their had has have".split())
PREDICTION_FIELDS = (
    *CLAIM_FIELDS, "true_label", "predicted_label", "support_score",
    "best_tfidf_similarity", "best_evidence_chunk_index", "best_evidence_chunk",
    "retrieved_chunk_count",
)


def tokens(text):
    return {word.casefold() for word in WORD.findall(text)} - STOP


def quantity_features(text):
    numbers = {match.group().replace(",", "") for match in NUMBER.finditer(text)}
    numbers.update(NUMBER_WORDS[word] for word in tokens(text) & NUMBER_WORDS.keys())
    words = [word.group().casefold().replace(",", "") for word in ANCHOR.finditer(text)]
    anchors = set()
    for index, word in enumerate(words):
        number = NUMBER_WORDS.get(word, word)
        if not NUMBER.fullmatch(number):
            continue
        for direction in (-1, 1):
            for distance in (1, 2):
                neighbor = index + direction * distance
                if 0 <= neighbor < len(words):
                    anchor = words[neighbor]
                    if anchor.isalpha() and anchor not in STOP and anchor not in NUMBER_WORDS:
                        anchors.add((number, anchor))
    return numbers, anchors


@lru_cache(maxsize=8192)
def describe(text):
    numbers, anchors = quantity_features(text)
    month_days = {
        (MONTH_IDS[(first_month or second_month).lower()], int(first_day or second_day))
        for first_month, first_day, second_day, second_month in MONTH_DAY.findall(text)
    }
    month_days.update((int(month), int(day)) for _, month, day in ISO_DATE.findall(text))
    return {
        "numbers": numbers,
        "anchors": anchors,
        "years": set(YEAR.findall(text)),
        "month_days": month_days,
        "tokens": tokens(text),
        "names": {part.casefold() for name in NAME.findall(text) for part in name.split()},
        "negation": bool(NEGATION.search(text)),
        "unknown": bool(UNKNOWN.search(text)),
        "null_fields": [tokens(re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", key))
                        for key in NULL_FIELD.findall(text)],
    }


def coverage(expected, observed):
    return len(expected & observed) / len(expected) if expected else 0.0


def compare(claim, evidence):
    features = {}
    for kind in ("numbers", "anchors", "years", "month_days", "names", "tokens"):
        expected, observed = claim[kind], evidence[kind]
        features[f"{kind}_coverage"] = coverage(expected, observed)
        if kind != "tokens":
            features[f"{kind}_unmatched"] = float(bool(expected - observed))
    features["negation_mismatch"] = float(claim["negation"] != evidence["negation"])
    features["evidence_unknown"] = float(evidence["unknown"])
    features["specific_date_with_unknown"] = float(
        bool(claim["month_days"] or claim["years"]) and evidence["unknown"]
    )
    features["mentioned_null_field"] = float(any(
        field and field <= claim["tokens"] for field in evidence["null_fields"]
    ))
    return features


def feature_rows(claims, retrieved):
    rows = []
    for claim, evidence in zip(claims, retrieved, strict=True):
        description = describe(claim["claim_text"])
        strongest = max(evidence, key=lambda item: item[2].entailment)
        row = {
            "best_tfidf_similarity": claim["max_similarity"],
            "retrieved_chunk_fraction": len(evidence) / 3,
            "nli_entailment": strongest[2].entailment,
            "nli_contradiction": strongest[2].contradiction,
            "nli_neutral": strongest[2].neutral,
            "nli_max_contradiction": max(scores.contradiction for _, _, scores in evidence),
            "nli_max_neutral": max(scores.neutral for _, _, scores in evidence),
            "claim_negation": float(description["negation"]),
            "claim_unknown": float(description["unknown"]),
            **{f"claim_has_{kind}": float(bool(description[kind]))
               for kind in ("numbers", "years", "month_days", "names")},
        }
        comparisons = [compare(description, describe(text)) for _, text, _ in evidence]
        row.update({f"best_entailment_{key}": value for key, value in
                    compare(description, describe(strongest[1])).items()})
        for key in comparisons[0]:
            values = [pair[key] for pair in comparisons]
            row[f"top3_min_{key}"] = min(values)
            row[f"top3_max_{key}"] = max(values)
        rows.append(row)
    describe.cache_clear()
    return rows


def evaluate_report(test, baselines, best, args, classifier, feature_encoder):
    lines = [
        "RAGTruth hybrid evidence-grounded detector",
        f"NLI checkpoint: {MODEL_ID}; CPU threads: {args.threads}; batch: {args.batch_size}.",
        "L2 logistic regression and feature encoding fitted on training claims only.",
        "Validation maximizes unsupported F1. Ties use the smallest cutoff.",
        "Prediction is unsupported when support (1 minus P(unsupported)) is below the cutoff.",
        f"Best validation threshold: {best['threshold']!r}",
        f"Best validation F1: {best['f1']:.6f}",
        "Retrieval and all factual cues use only the claim and its own top-three source chunks.",
        "Factual cues are approximate. Surface mismatches and chunk-level unknowns are soft features.",
        "The evidence displayed is the strongest-entailment chunk; the classifier uses all three.",
        "ROC-AUC uses negative support/similarity. Matrix rows are true, columns predicted, supported first.",
    ]
    datasets = (*baselines, ("Hybrid", test, "support_score"))
    for scope in ("Overall", *TASKS):
        lines.extend([
            "", scope,
            f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'ROC-AUC':>10}  Confusion matrix",
        ])
        for name, rows, field in datasets:
            subset = rows if scope == "Overall" else [row for row in rows if row["task_type"] == scope]
            metrics = evaluate(subset, field)
            values = " ".join(f"{format_metric(metrics[key]):>10}" for key in METRICS)
            matrix = f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]"
            lines.append(f"{name:<10} {values}  {matrix}")
    lines.extend(["", "Unsupported log-odds coefficients", f"Intercept: {classifier.intercept_[0]:+.6f}"])
    lines.extend(f"{name}: {weight:+.6f}" for name, weight in
                 zip(feature_encoder.get_feature_names_out(), classifier.coef_[0], strict=True))
    return "\n".join(lines) + "\n"


def explain_errors(test, feature_matrix, retrieved, encoder, classifier):
    lines = ["Contributions are feature value times coefficient, in unsupported log odds.",
             f"Intercept: {classifier.intercept_[0]:+.6f}."]
    names = encoder.get_feature_names_out()
    for title, true_label in (("False positives", SUPPORTED), ("False negatives", UNSUPPORTED)):
        errors = [index for index, claim in enumerate(test)
                  if claim["true_label"] == true_label and claim["predicted_label"] != true_label]
        lines.append(f"\n{title} (showing {min(5, len(errors))} of {len(errors):,})")
        for index in errors[:5]:
            claim = test[index]
            values = feature_matrix[index]
            contributions = sorted(zip(names, values, values * classifier.coef_[0]),
                                   key=lambda item: -abs(item[2]))
            lines.extend([
                f"\n{claim['claim_id']} ({claim['task_type']}, source {claim['source_id']}) "
                f"support={claim['support_score']:.6f}",
                f"Claim: {claim['claim_text']}",
                *(f"Chunk {chunk_index} (E={scores.entailment:.4f}, C={scores.contradiction:.4f}, "
                  f"N={scores.neutral:.4f}): {text}"
                  for chunk_index, text, scores in retrieved[index]),
                "Largest feature contributions:",
                *(f"  {name}={value:.4f}, contribution={delta:+.4f}"
                  for name, value, delta in contributions[:8]),
            ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Local snapshot of the frozen TinyBERT checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and threads must be positive")

    torch.set_num_threads(args.threads)
    checkpoint = args.model_path or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).cpu().eval()
    label_ids = model_label_ids(model)
    seen_sources = set()
    print("Loading frozen training split and fitting TF-IDF.", flush=True)
    train, train_chunks = load_split("train", seen_sources)
    vectorizer = fit_vectorizer(train_chunks)

    def extract(split, claims, chunks):
        retrieved = verify_split(split, vectorizer, claims, chunks, model,
                                 tokenizer, label_ids, args.batch_size)
        print(f"{split}: extracting factual features.", flush=True)
        return feature_rows(claims, retrieved), retrieved

    train_rows, train_evidence = extract("train", train, train_chunks)
    encoder = DictVectorizer(sparse=False)
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(encoder.fit_transform(train_rows),
                   [claim["true_label"] == UNSUPPORTED for claim in train])
    del train, train_chunks, train_rows, train_evidence

    def score(claims, rows):
        matrix = encoder.transform(rows)
        supported_column = list(classifier.classes_).index(False)
        for claim, probability in zip(claims, classifier.predict_proba(matrix)[:, supported_column]):
            claim["support_score"] = float(probability)
        return matrix

    validation, validation_chunks = load_split("validation", seen_sources)
    validation_rows, validation_evidence = extract("validation", validation, validation_chunks)
    score(validation, validation_rows)
    del validation_rows, validation_chunks, validation_evidence
    thresholds = validation_thresholds(validation, "support_score")
    best = max(thresholds, key=lambda row: row["f1"])
    cutoff = best["threshold"]
    print(f"Frozen validation threshold: {cutoff!r} (F1 {best['f1']:.6f}).", flush=True)

    test, test_chunks = load_split("test", seen_sources)
    baselines = [
        ("TF-IDF", read_baseline_predictions(test, "tfidf_test_predictions.csv"), "max_similarity"),
        ("TinyBERT", read_baseline_predictions(test, "nli_test_predictions.csv"), "support_score"),
    ]
    test_rows, evidence = extract("test", test, test_chunks)
    matrix = score(test, test_rows)
    for claim in test:
        claim.update(
            predicted_label=UNSUPPORTED if claim["support_score"] < cutoff else SUPPORTED,
            best_tfidf_similarity=claim["max_similarity"],
        )

    report = evaluate_report(test, baselines, best, args, classifier, encoder)
    errors = explain_errors(test, matrix, evidence, encoder, classifier)
    write_csv(RESULTS_DIR / "hybrid_validation_thresholds.csv", thresholds, list(best))
    write_csv(RESULTS_DIR / "hybrid_test_predictions.csv",
              ({key: claim[key] for key in PREDICTION_FIELDS} for claim in test), PREDICTION_FIELDS)
    (RESULTS_DIR / "hybrid_metrics.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "hybrid_error_examples.txt").write_text(errors, encoding="utf-8")
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
