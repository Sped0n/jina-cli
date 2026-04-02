---
urls:
  - https://github.com/jina-ai/cli#readme
---

# Search and Read

## Goals

- Use `jina search` for discovery and `jina read` for extraction.
- Keep web-oriented commands pipe-friendly.

## Guidance

- Use `jina search QUERY` for general web search.
- Reach for domain flags like `--arxiv`, `--ssrn`, `--images`, or `--blog` when the source type matters.
- Use `jina read URL` to extract clean markdown from web pages.
- Add `--links` or `--images` only when you need richer extraction output.
- Use `jina pdf` for PDF-specific extraction rather than `jina read` when figures, tables, or equations matter.
- Prefer `--json` if the results will be filtered or fed into a follow-up command.

## Examples

General web search:

```bash
jina search "what is BERT"
```

Target arXiv specifically:

```bash
jina search --arxiv "attention mechanism" -n 10
```

Read a page as markdown:

```bash
jina read https://example.com
```

Read multiple URLs from stdin:

```bash
cat urls.txt | jina read
```

Extract from a PDF:

```bash
jina pdf https://arxiv.org/pdf/2301.12345 --type figure,table
```

Get structured search results:

```bash
jina search "embeddings" --json | jq -r '.results[].url'
```
