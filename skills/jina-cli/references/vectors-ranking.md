# Vectors and Ranking

## Goals

- Choose the right vector-oriented subcommand for the task.
- Keep ranking and classification flows batch-friendly.

## Guidance

- Use `jina embed` to generate embeddings for one or more texts.
- Use `jina rerank QUERY` when the candidate documents already exist and come from stdin.
- Use `jina classify TEXT --labels ...` for lightweight label assignment.
- Use `jina dedup` to remove near-duplicate text items from stdin.
- Prefer stdin when processing many documents or piping from `jina search`.
- Add `--json` when post-processing scores, labels, or vectors programmatically.

## Examples

Generate embeddings:

```bash
jina embed "hello world"
```

Batch embeddings from stdin:

```bash
cat texts.txt | jina embed --json
```

Rerank a document stream:

```bash
cat docs.txt | jina rerank "machine learning"
```

Search and rerank together:

```bash
jina search "AI" | jina rerank "embeddings" --top-n 5
```

Classify with explicit labels:

```bash
jina classify "I love this product" --labels positive,negative,neutral
```

Deduplicate streamed items:

```bash
cat items.txt | jina dedup
```
