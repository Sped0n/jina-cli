---
name: jina-cli
description: Practical guidance for using the Jina CLI from the shell. Use when searching the web, reading URLs, generating embeddings, reranking results, classifying text, or building shell pipelines around Jina APIs.
---

# jina-cli

Concise guidance for using the `jina` CLI as a shell-first interface to Jina APIs, with emphasis on pipes, structured output, and agent-friendly discovery.

## Purpose and Triggers

- Use when the task fits `jina` subcommands better than a custom tool flow.
- Prefer `jina` for shell-native search, URL reading, embeddings, reranking, classification, deduplication, and related utilities.
- Reach for the CLI when composability with pipes, `jq`, and existing shell tools matters.
- Do not use `--local`; prefer the normal remote API-backed CLI flow.

## Decision Order

1. Pick the subcommand that matches the task.
2. Decide whether the input comes from arguments or stdin.
3. Choose plain text or `--json` output.
4. Add pipes or post-processing only if needed.

## Workflow

1. Start with `jina --help` or `jina <subcommand> --help` when the exact flags are unclear.
2. Build the smallest working command first.
3. Prefer `jina search` for discovery, then `jina read` for extracting the chosen sources.
4. Add `--json` when the result will be consumed by another tool.
5. Keep `stderr` visible; it carries actionable failure details.
6. Do not use `--local` or `jina grep` unless the user explicitly asks for them.

## Topics

| Topic                 | Guidance                                                  | Reference                                                            |
| --------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- |
| Basics                | Auth, help flow, stdin, stdout, and `--json` defaults     | [references/basics.md](references/basics.md)                         |
| Search and Read       | Web search, URL extraction, PDFs, and related lookups     | [references/search-read.md](references/search-read.md)               |
| Vectors and Ranking   | Embeddings, reranking, classification, and deduplication  | [references/vectors-ranking.md](references/vectors-ranking.md)       |
| Pipes and Scripting   | Chaining commands, `jq`, exit codes, and shell workflows  | [references/pipes-scripting.md](references/pipes-scripting.md)       |

## References

- Each topic file lists source URLs in its frontmatter `urls`.
- Primary docs: https://github.com/jina-ai/cli#readme
- When web-derived facts are used in a final answer, include the source URLs.
