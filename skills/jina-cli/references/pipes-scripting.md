---
urls:
  - https://github.com/jina-ai/cli#readme
---

# Pipes and Scripting

## Goals

- Preserve Unix composability.
- Make automation robust around output format and exit codes.

## Guidance

- Prefer plain text output when feeding another `jina` subcommand directly.
- Prefer `--json` when the next stage is `jq`, Python, or another structured consumer.
- Use exit codes instead of string matching for control flow.
- Keep `stderr` separate from stdout in scripts; stdout is data, stderr is diagnostics.
- Build chains from small commands instead of one monolithic shell line when debugging.

## Examples

Chain search into rerank:

```bash
jina search "transformer models" | jina rerank "efficient inference"
```

Expand a query, then search the first expansion:

```bash
jina expand "climate change" | head -1 | xargs -I {} jina search "{}"
```

Use JSON with `jq`:

```bash
jina read https://example.com --json | jq -r '.data.content'
```

Branch on exit status:

```bash
jina search "query" && echo success || echo "failed with $?"
```
