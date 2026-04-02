# Basics

## Goals

- Pick the right subcommand quickly.
- Keep commands composable and easy to inspect.

## Guidance

- Treat `jina` as the primary interface; prefer `jina <subcommand>` over ad hoc API requests when the CLI already covers the task.
- Use `jina --help` first, then `jina <subcommand> --help` for flag details.
- Assume `JINA_API_KEY` is required for most commands that call Jina services.
- Use the standard remote CLI behavior for all commands.
- Prefer positional arguments for simple one-shot commands and stdin for batch flows.
- Use `--json` whenever the output will be piped into `jq` or another parser.
- Use the global `--timeout SECONDS` flag when a request needs more or less time than the built-in default.
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

````bash
jina search "BERT" --json | jq '.results[0].url'

Override the default timeout globally:

```bash
jina --timeout 45 search "complex query"
````

```

```
