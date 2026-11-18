import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Iterator

from explore_context import (
    PROJECT_ROOT,
    RESPONSES_PATH,
    SOURCES_PATH,
    build_source_lookup,
    read_jsonl,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SUMMARY_PATH = PROJECT_ROOT / "results" / "claim_dataset_summary.txt"
SPLITS = ("train", "validation", "test")
SENTENCE_END = re.compile(r"""[.!?]+["'”’\)\]\}]*?(?=\s|$)""")
NONTERMINAL_PERIOD = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|St|Mt|Gen|Sen|Rep|Gov|Lt|Sgt|Capt|"
    r"e\.g|i\.e|vs|[A-Z]|(?:[A-Z]\.)+[A-Z])\.$"
)
LIST_NUMBER = re.compile(r"\s*(?:\*\*)?\d+\.")


def sentence_ranges(text: str) -> Iterator[tuple[int, int]]:
    # Treat each nonempty line as a unit before splitting on sentence punctuation.
    for line in re.finditer(r"[^\r\n]+", text):
        start = line.start()
        for match in SENTENCE_END.finditer(text, start, line.end()):
            end = match.end()
            if end < line.end() and match.group() == ".":
                prefix = text[start:end]
                if NONTERMINAL_PERIOD.search(prefix) or LIST_NUMBER.fullmatch(prefix):
                    continue
            while start < end and text[start].isspace():
                start += 1
            yield start, end
            start = end
        end = line.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end


def assign_splits(responses: list[dict]) -> tuple[dict, str]:
    split_by_source = {}
    for response in responses:
        source_id, split = response["source_id"], response["split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected split {split!r} for response {response['id']}")
        if split_by_source.setdefault(source_id, split) != split:
            raise ValueError(f"Source {source_id!r} occurs in multiple input splits")

    if "validation" in split_by_source.values():
        return split_by_source, "Preserved existing source splits."

    # Hash ordering makes the holdout reproducible regardless of input row order.
    train_sources = sorted(
        (source_id for source_id, split in split_by_source.items() if split == "train"),
        key=lambda source_id: hashlib.sha256(json.dumps(source_id).encode()).digest(),
    )
    count = max(1, len(train_sources) // 10) if len(train_sources) > 1 else 0
    for source_id in train_sources[:count]:
        split_by_source[source_id] = "validation"
    return split_by_source, (
        f"Validation holds {count} of {len(train_sources)} original training sources "
        "using deterministic SHA-256 ordering. Original test split preserved."
    )


def validate_labels(text: str, labels: list[dict], response_id: Any) -> list[str]:
    mismatches = []
    for index, label in enumerate(labels):
        start, end = label.get("start"), label.get("end")
        location = f"Response {response_id}, label {index}, range [{start}, {end})"
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError(f"{location}: invalid offsets for text of length {len(text)}")
        expected = label.get("text")
        if not isinstance(expected, str):
            mismatches.append(f"{location}: label text unavailable for comparison")
        elif text[start:end] != expected:
            mismatches.append(
                f"{location}: label text {expected!r} differs from slice {text[start:end]!r}"
            )
    return mismatches


def build_claims(
    response: dict, source: dict, split: str
) -> Iterator[dict]:
    text = response["response"]
    for index, (start, end) in enumerate(sentence_ranges(text), start=1):
        # RAGTruth offsets are half-open, so adjacent ranges do not overlap.
        overlaps = [
            label for label in response["labels"]
            if start < label["end"] and label["start"] < end
        ]
        yield {
            "response_id": response["id"],
            "source_id": response["source_id"],
            "split": split,
            "model": response["model"],
            "task_type": source["task_type"],
            "claim_id": f"{response['id']}:{index}",
            "claim_text": text[start:end],
            "claim_start": start,
            "claim_end": end,
            "claim_label": "unsupported" if overlaps else "supported",
            "hallucination_spans": overlaps,
            "source_info": source["source_info"],
            "prompt": source["prompt"],
            "has_null_based_hallucination": any(
                label.get("due_to_null") is True for label in overlaps
            ),
            "hallucination_types": sorted({label["label_type"] for label in overlaps}),
        }


def main() -> None:
    sources = build_source_lookup(read_jsonl(SOURCES_PATH))
    responses = [
        response for response in read_jsonl(RESPONSES_PATH)
        if response.get("quality") == "good"
    ]
    split_by_source, split_description = assign_splits(responses)
    claims_by_split = defaultdict(list)
    task_counts = defaultdict(Counter)
    claim_type_counts = Counter()
    span_type_counts = Counter()
    claim_counts = Counter()
    examples = {}
    seen_responses = set()
    mismatches = []
    null_spans = 0
    null_claims = 0

    for response in responses:
        response_id, source_id = response["id"], response["source_id"]
        if response_id in seen_responses:
            raise ValueError(f"Duplicate good-quality response ID: {response_id}")
        seen_responses.add(response_id)
        if source_id not in sources:
            raise ValueError(f"Response {response_id} has no source for ID {source_id!r}")
        text, labels = response["response"], response["labels"]
        if not isinstance(text, str) or not isinstance(labels, list):
            raise ValueError(f"Response {response_id} requires text and a label list")

        mismatches.extend(validate_labels(text, labels, response_id))
        span_type_counts.update(label["label_type"] for label in labels)
        null_spans += sum(label.get("due_to_null") is True for label in labels)
        split = split_by_source[source_id]
        for claim in build_claims(response, sources[source_id], split):
            claims_by_split[split].append(claim)
            status = claim["claim_label"]
            claim_counts[status] += 1
            task_counts[claim["task_type"]][status] += 1
            claim_type_counts.update(claim["hallucination_types"])
            null_claims += claim["has_null_based_hallucination"]
            examples.setdefault(status, claim)

    total = sum(claim_counts.values())
    rate = claim_counts["unsupported"] / total if total else 0
    summary = [
        "RAGTruth Claim Dataset Summary",
        f"Total good-quality responses: {len(responses):,}",
        f"Total generated claims: {total:,}",
        f"Supported claims: {claim_counts['supported']:,}",
        f"Unsupported claims: {claim_counts['unsupported']:,}",
        f"Claim-level hallucination rate: {rate:.2%}",
        f"Null-based hallucinations (original spans): {null_spans:,}",
        f"Claims overlapping null-based hallucinations: {null_claims:,}",
        f"Span text validation issues: {len(mismatches):,}",
        *mismatches,
        "",
        split_description,
        "Claims by split:",
        *(f"- {split}: {len(claims_by_split[split]):,}" for split in SPLITS),
        "",
        "Claims by task type:",
    ]
    for task, counts in sorted(task_counts.items()):
        summary.append(
            f"- {task}: {sum(counts.values()):,} total, "
            f"{counts['supported']:,} supported, {counts['unsupported']:,} unsupported"
        )
    summary.extend(["", "Hallucination types (original spans and overlapping claims):"])
    for label_type, count in sorted(span_type_counts.items()):
        summary.append(
            f"- {label_type}: {count:,} spans, {claim_type_counts[label_type]:,} claims"
        )
    for status in ("supported", "unsupported"):
        example = examples.get(status)
        summary.extend(["", f"{status.title()} claim example:"])
        if example is None:
            summary.append("No example available.")
            continue
        summary.extend(
            f"{key}: {json.dumps(example[key], ensure_ascii=False)}"
            for key in ("claim_id", "claim_start", "claim_end", "claim_text", "hallucination_spans")
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (OUTPUT_DIR / f"{split}_claims.jsonl").open("w", encoding="utf-8") as output:
            for claim in claims_by_split[split]:
                output.write(json.dumps(claim, ensure_ascii=False, separators=(",", ":")) + "\n")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    for mismatch in mismatches:
        print(f"Warning: {mismatch}", file=sys.stderr)
    print(f"Built {total:,} claims from {len(responses):,} good-quality responses.")
    print(f"Unsupported: {claim_counts['unsupported']:,} ({rate:.2%}).")
    print(f"Span text validation issues: {len(mismatches):,}.")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
