# RAG Hallucination Detection

This project investigates whether NLP models can detect unsupported claims in retrieval-augmented generation (RAG) responses.

## Research Question

Can semantic similarity and Natural Language Inference models identify hallucinated claims in LLM-generated answers when compared with retrieved source evidence?

## Dataset

The project uses the RAGTruth dataset, which contains retrieved source information, prompts, LLM-generated responses, and human-annotated hallucination spans.

## Current Progress

- Downloaded and inspected the RAGTruth dataset
- Implemented safe text normalization
- Preserved case, punctuation, numbers, dates, named entities, and negation
- Generated initial dataset statistics

## Initial Results

- Source records: 2,965
- LLM responses: 17,790
- Good-quality responses: 17,617
- Responses containing unsupported spans: 7,641
- Initial hallucination rate: 43.37%

## Next Steps

1. Create response-level supported and unsupported labels
2. Build a TF-IDF similarity baseline
3. Evaluate a pretrained Natural Language Inference model
4. Analyze errors involving irrelevant evidence and subtle factual changes

# Data

This project uses the RAGTruth dataset.

Download the dataset from:

https://github.com/ParticleMedia/RAGTruth

The raw `.jsonl` files are intentionally excluded from this repository.
They can be found in the downloaded RAGTruth repository under:

RAGTruth/dataset/