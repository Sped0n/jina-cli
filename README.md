# jina-cli

A small personal CLI for the Jina APIs I use most often.

This repo is intentionally narrow: a shell-first tool for `read`, `search`, embeddings, reranking, classification, deduplication, screenshots, BibTeX lookup, PDF extraction, and related utilities.

The interface is optimized for direct terminal use and simple automation. Commands are pipe-friendly, support `--help` for discovery, and now share a global `--timeout` flag for slow requests.

## Install

```bash
pip install jina-cli
# or
uv pip install jina-cli
```

Set your API key:
```bash
export JINA_API_KEY=your-key-here
# Get one at https://jina.ai/?sui=apikey
```

## Commands

| Command | Description |
|---------|-------------|
| `jina read URL` | Extract clean markdown from web pages |
| `jina search QUERY` | Web search (also --arxiv, --ssrn, --images, --blog) |
| `jina embed TEXT` | Generate embeddings |
| `jina rerank QUERY` | Rerank documents from stdin by relevance |
| `jina classify TEXT` | Classify text into labels |
| `jina dedup` | Deduplicate text from stdin |
| `jina screenshot URL` | Capture screenshot of a URL |
| `jina bibtex QUERY` | Search BibTeX citations (DBLP + Semantic Scholar) |
| `jina expand QUERY` | Expand a query into related queries |
| `jina pdf URL` | Extract figures/tables/equations from PDFs |
| `jina datetime URL` | Guess publish/update date of a URL |
| `jina primer` | Context info (time, location, network) |

## Global options

- `--timeout FLOAT` overrides the default HTTP timeout for all network-backed commands.

## Pipes

The point of a CLI is composability. Every command reads from stdin and writes to stdout.

```bash
# Search and rerank
jina search "transformer models" | jina rerank "efficient inference"

# Read multiple URLs
cat urls.txt | jina read

# Search, deduplicate results
jina search "attention mechanism" | jina dedup

# Chain searches
jina expand "climate change" | head -1 | xargs -I {} jina search "{}"

# Get BibTeX for arXiv results
jina search --arxiv "BERT" --json | jq -r '.results[].title' | head -3

# Allow a longer request window for a slow page
jina --timeout 60 read https://example.com/large-page
```

## Usage

### Read web pages

```bash
jina read https://example.com
jina read https://example.com --links --images
echo "https://example.com" | jina read
```

### Search

```bash
jina search "what is BERT"
jina search --arxiv "attention mechanism" -n 10
jina search --ssrn "corporate governance"
jina search --images "neural network diagram"
jina search --blog "embeddings"
jina search "AI news" --time d          # past day
jina search "LLMs" --gl us --hl en     # US, English
```

### Embed

```bash
jina embed "hello world"
jina embed "text1" "text2" "text3"
cat texts.txt | jina embed
jina embed "hello" --model jina-embeddings-v5-text-small --task retrieval.query
```

### Rerank

```bash
cat docs.txt | jina rerank "machine learning"
jina search "AI" | jina rerank "embeddings" --top-n 5
```

### Classify

```bash
jina classify "I love this product" --labels positive,negative,neutral
echo "stock prices rose sharply" | jina classify --labels business,sports,tech
cat texts.txt | jina classify --labels cat1,cat2,cat3 --json
```

### Deduplicate

```bash
cat items.txt | jina dedup
cat items.txt | jina dedup -k 10
```

### Screenshot

```bash
jina screenshot https://example.com                        # prints screenshot URL
jina screenshot https://example.com -o page.png            # saves to file
jina screenshot https://example.com --full-page -o page.jpg
```

### BibTeX

```bash
jina bibtex "attention is all you need"
jina bibtex "transformer" --author Vaswani --year 2017
```

### PDF extraction

```bash
jina pdf https://arxiv.org/pdf/2301.12345
jina pdf 2301.12345                        # arXiv ID shorthand
jina pdf https://example.com/paper.pdf --type figure,table
```

## JSON output

Every command supports `--json` for structured output, useful for piping to `jq`:

```bash
jina search "BERT" --json | jq '.results[0].url'
jina read https://example.com --json | jq '.data.content'
```

## Retry behavior

Network-backed requests retry only on transient failures:

- network timeouts and connection errors
- `429` rate limits
- `5xx` responses returned by Jina endpoints

They do not retry normal client-side `4xx` errors or failures from non-Jina targets.

Backoff is exponential: `0.5s -> 1s -> 2s`, capped at `30s`.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | User/input error (missing args, bad input, missing API key) |
| 2 | API/server error (network, timeout, server error) |
| 130 | Interrupted (Ctrl+C) |

Useful for scripting and agent workflows:

```bash
jina search "query" && echo "success" || echo "failed with $?"
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `JINA_API_KEY` | API key for Jina services (required for most commands) |

## Automation

Shell automation can call the CLI directly:

```python
result = run(command="jina search 'transformer architecture'")
result = run(command="jina read https://arxiv.org/abs/2301.12345")
result = run(command="jina search 'AI' | jina rerank 'embeddings'")
```

The CLI surface is intentionally small enough to discover from `jina --help` and per-command help.

## Design principles

Inspired by [CLI is All Agents Need](https://x.com/yan5xu/status/2031947154911351159):

- **One tool, not twenty.** A single `run(command="jina search ...")` replaces a sprawling tool catalog. Less tool selection overhead, more problem solving.
- **Unix pipes are the composition model.** `stdout` is data, `stderr` is diagnostics. Commands chain with `|`, `&&`, `||`. No SDK needed.
- **Progressive `--help` for self-discovery.** Layer 0: command list. Layer 1: usage + examples. Layer 2: full options. The agent fetches only what it needs, saving context budget.
- **Error messages that course-correct.** Every error says what went wrong and exactly how to fix it. One bad command should not cost more than one retry.
- **`stderr` is the agent's most important channel.** When a command fails, `stderr` carries the fix. Never discard it. Never mix it with data.
- **Consistent output format.** Same structure every time so the agent learns once, not every time. `--json` for structured, plain text for pipes.
- **Meaningful exit codes.** `0` success, `1` user error, `2` API error, `130` interrupted. Scripts and agents branch on these, not on parsing error strings.
- **Layer 1 is raw Unix, Layer 2 is for LLM cognition.** Pipe internals stay pure (no metadata, no truncation). Formatting and context only at the final output boundary.

## License

Apache-2.0
