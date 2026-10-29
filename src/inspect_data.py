import json
from collections import Counter
from pathlib import Path

from preprocess import normalize_text


DATASET_DIR = Path("../RAGTruth/dataset")
RESPONSES_PATH = DATASET_DIR / "response.jsonl"
SOURCES_PATH = DATASET_DIR / "source_info.jsonl"
RESULTS_DIR = Path("results")


def read_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    responses = read_jsonl(RESPONSES_PATH)
    sources = read_jsonl(SOURCES_PATH)

    good_responses = [
        item for item in responses
        if item.get("quality") == "good"
    ]

    hallucinated = [
        item for item in good_responses
        if len(item.get("labels", [])) > 0
    ]

    model_counts = Counter(item["model"] for item in good_responses)
    task_counts = Counter(item["task_type"] for item in sources)

    normalized_changed = sum(
        normalize_text(item["response"]) != item["response"]
        for item in good_responses
    )

    summary = [
        "RAGTruth Dataset Inspection",
        "=" * 30,
        f"Total source records: {len(sources)}",
        f"Total response records: {len(responses)}",
        f"Good-quality responses used: {len(good_responses)}",
        f"Responses with at least one unsupported span: {len(hallucinated)}",
        f"Hallucination rate: {len(hallucinated) / len(good_responses):.2%}",
        f"Responses changed by safe normalization: {normalized_changed}",
        "",
        "Responses by generating model:",
    ]

    for model, count in model_counts.most_common():
        summary.append(f"- {model}: {count}")

    summary.append("")
    summary.append("Source records by task type:")

    for task, count in task_counts.most_common():
        summary.append(f"- {task}: {count}")

    output_path = RESULTS_DIR / "dataset_summary.txt"
    output_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()