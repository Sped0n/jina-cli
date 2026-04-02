---
urls:
  - https://github.com/jina-ai/cli#readme
  - https://pypi.org/project/jina-cli/
---

# Basics

## Goals

- Pick the right subcommand quickly.
- Keep commands composable and easy to inspect.

## Guidance

- Treat `jina` as the primary interface; prefer `jina <subcommand>` over ad hoc API requests when the CLI already covers the task.
- Use `jina --help` first, then `jina <subcommand> --help` for flag details.
- Assume `JINA_API_KEY` is required for most remote commands unless you are explicitly using local mode.
- Do not use `--local`; stick to the normal remote CLI behavior unless the user explicitly asks for local execution.
- Prefer positional arguments for simple one-shot commands and stdin for batch flows.
- Use `--json` whenever the output will be piped into `jq` or another parser.
- Keep `stderr` intact; Jina CLI uses it for diagnostics and recovery hints.

## Examples

Show top-level help:

```bash
jina --help
```

Show help for a specific command:

```bash
jina search --help
```

Read input from arguments:

```bash
jina embed "hello world"
```

Read input from stdin:

```bash
printf '%s\n' "https://example.com" | jina read
```

Structured output for downstream parsing:

```bash
jina search "BERT" --json | jq '.results[0].url'
```
