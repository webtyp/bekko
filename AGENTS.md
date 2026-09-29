# Agent Guide — `webtyp/bekko`

Constraints for agents working on this library. **Read this before any change.**
The current work order, when one exists, is [docs/PLAN.md](docs/PLAN.md). The master index is
[`retrieval/docs/SEMANTIC_SEARCH_MASTER_PLAN.md`](https://github.com/webtyp/retrieval/blob/main/docs/SEMANTIC_SEARCH_MASTER_PLAN.md).

## What this library is

One implementation of the `embed.Embedder` contract, for exactly one model:
`bekko-embedding-v1-a8m`. Everything specific to that model lives here: special token ids,
pre-tokenizer, weight names, and the Matryoshka truncation to 128 dimensions followed by
renormalization. It **composes** `tokenizer`, `weights` and `encoder`. It never re-implements them.

It never downloads or caches the model. The caller passes the artifact bytes already read.

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
```

## Never import these

| Never | Use instead | Why |
|---|---|---|
| `strings`, `fmt`, `errors`, `strconv` | `webtyp.com/fmt` | isomorphism + TinyGo size |
| `context` (stdlib) | `webtyp.com/context` | |
| `encoding/json` | nothing | reflection JSON costs ~1 MB of wasm |
| `map[K]V` | a slice | TinyGo's map runtime is a size tax |
| `os`, `log`, `net/http` | inject it | a library never touches the process environment |

## Memory shape — not optional

The token embedding table (256 000 × 384 int8) is **never dequantized in full**. It is read
row by row for the tokens of each call. The 4 transformer layers are small enough to dequantize
once and keep.

## Common mistakes to avoid

- Skipping renormalization after truncating to 128 dimensions. The first 128 components of a
  unit vector are not unit length.
- Trusting a synthetic fixture for the reference test. It compares against the real model's
  output, and nothing else proves the port.
