import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT.parent / "RAGTruth" / "dataset"
RESPONSES_PATH = DATASET_DIR / "response.jsonl"
SOURCES_PATH = DATASET_DIR / "source_info.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "context_linkage_examples.txt"
EXAMPLE_COUNT = 5
MAX_DISPLAY_TEXT_LENGTH = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            records.append(record)
    return records


def build_source_lookup(
    source_records: list[dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    source_lookup = {}
    for record in source_records:
        source_id = record.get("source_id")
        if source_id is None:
            continue
        if source_id in source_lookup:
            raise ValueError(f"Duplicate source_id in source records: {source_id}")
        source_lookup[source_id] = record
    return source_lookup


def select_examples(
    responses: list[dict[str, Any]],
    source_lookup: dict[Any, dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = []
    seen_source_ids = set()
    for response in responses:
        source_id = response.get("source_id")
        if response.get("quality") != "good" or source_id not in source_lookup:
            continue
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        candidates.append((response, source_lookup[source_id]))

    if len(candidates) < EXAMPLE_COUNT:
        raise ValueError(
            f"Found only {len(candidates)} linked good-quality responses with "
            f"distinct source IDs; need {EXAMPLE_COUNT}."
        )

    selected = candidates[:EXAMPLE_COUNT]
    if not any(isinstance(source.get("source_info"), dict) for _, source in selected):
        structured_example = next(
            (
                pair
                for pair in candidates[EXAMPLE_COUNT:]
                if isinstance(pair[1].get("source_info"), dict)
            ),
            None,
        )
        if structured_example is not None:
            selected[-1] = structured_example

    return selected


def truncate_display_text(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_DISPLAY_TEXT_LENGTH:
        remaining = len(value) - MAX_DISPLAY_TEXT_LENGTH
        return (
            value[:MAX_DISPLAY_TEXT_LENGTH]
            + f"\n...[truncated {remaining} characters for display]..."
        )
    if isinstance(value, dict):
        return {key: truncate_display_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [truncate_display_text(item) for item in value]
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, str):
        return truncate_display_text(value)
    return json.dumps(truncate_display_text(value), ensure_ascii=False, indent=2)


def collect_key_paths(value: Any, prefix: str = "") -> set[str]:
    key_paths = set()
    if isinstance(value, dict):
        for key, nested_value in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_paths.add(path)
            key_paths.update(collect_key_paths(nested_value, path))
    elif isinstance(value, list):
        for nested_value in value:
            if isinstance(nested_value, (dict, list)):
                key_paths.update(collect_key_paths(nested_value, f"{prefix}[*]"))
    return key_paths


def format_example(
    index: int,
    response: dict[str, Any],
    source: dict[str, Any],
) -> str:
    source_id = response.get("source_id")
    linked_source_id = source.get("source_id")
    if source_id != linked_source_id:
        raise ValueError(
            f"Link mismatch for response {response.get('id', 'Not available')}: "
            f"{source_id!r} != {linked_source_id!r}"
        )

    key_paths = sorted(collect_key_paths(source))
    key_list = "\n".join(f"- {key}" for key in key_paths) or "- None"
    labels = response.get("labels")
    if labels is None:
        labels = "Not available"
    else:
        labels = format_value(labels)

    sections = [
        f"Example {index}",
        "-" * 40,
        f"Response ID: {format_value(response.get('id'))}",
        f"Source ID: {format_value(source_id)}",
        f"Link verification: MATCH (response source_id == linked source_id)",
        f"Generating model: {format_value(response.get('model'))}",
        f"Dataset split: {format_value(response.get('split'))}",
        f"Task type: {format_value(source.get('task_type'))}",
        "Prompt/question:",
        format_value(source.get("prompt")),
        "Retrieved source/context content (source_info):",
        format_value(source.get("source_info")),
        "LLM-generated response:",
        format_value(response.get("response")),
        "Human hallucination labels/spans (labels):",
        labels,
        "All available linked source record key names (including nested keys):",
        key_list,
    ]
    return "\n".join(sections)


def main() -> None:
    responses = read_jsonl(RESPONSES_PATH)
    source_records = read_jsonl(SOURCES_PATH)
    source_lookup = build_source_lookup(source_records)
    examples = select_examples(responses, source_lookup)

    report = "\n\n".join(
        format_example(index, response, source)
        for index, (response, source) in enumerate(examples, start=1)
    )
    output = (
        "RAGTruth Response-to-Source Context Linkage Examples\n"
        + "=" * 54
        + "\n\n"
        + report
        + "\n"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"\nSaved examples to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
