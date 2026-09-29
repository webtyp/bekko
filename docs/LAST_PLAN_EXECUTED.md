---
PLAN: "feat: bekko — the bekko-embedding-v1-a8m embedder, moved out of webtyp/embed"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> **Blocked until `webtyp.com/encoder` v0.2.0 is published** (the rename of `webtyp/transformer`).
> `webtyp/embed` v0.4.0 (which deletes its copy) waits for this tag.

# Plan — `webtyp.com/bekko`: one embedding model, in its own repository

## 0. Context

`webtyp/embed` holds two things:

1. **the contract**: the `Embedder` interface (text in, vectors out, plus `CountTokens`) and a
   deterministic `MockEmbedder` for tests;
2. **one implementation**: `StaticEmbedder`, which runs the real model
   `bekko-embedding-v1-a8m` in the browser by composing `tokenizer` + `weights` + the encoder.

Because both live in one package, anything that only needs the interface (`vectordb`,
`agentmemory`, `retrieval`) pulls the tokenizer, the weight reader and the encoder into its
binary. The ecosystem rule is that a repository exposing a contract must not ship an
implementation. So the implementation moves here, to a repository named after the model it
runs. Database drivers in this ecosystem follow the same rule (`postgres`, `sqlt`, `indexdb`
implement `storage`).

This plan **copies** the implementation. Deleting it from `embed` is the next plan.

## Development rules (inline)

- Primary runtime: browser, TinyGo/WASM. Every file compiles under `GOOS=js GOARCH=wasm` and TinyGo.
- **Behaviour must not change**: the vectors were verified to cosine ≥ 0.999 against the real
  model. The token embedding table is **never dequantized in full** (256 000 × 384). Keep the
  row-by-row `dequantRow` exactly as it is.
- **Never import:** `fmt`, `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), `context`
  (use `webtyp.com/context`), `encoding/json`, `map[K]V`, `os`, `log`, `net/http`. `math` is allowed.
- This package never downloads or caches the model. It receives the bytes already read.
- Tests: `testing` only. Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** **LangChain** ships one embeddings integration per provider/model
   (`OpenAIEmbeddings`, `HuggingFaceEmbeddings`) against a single `Embeddings` interface.
   **sentence-transformers** loads each model as its own artifact behind one `encode()` API.
   **Go `database/sql`** drivers live in their own modules against one interface. In all
   three, the contract has no model code.
2. **Novice-name test.** `bekko.New(bekko.Config{ArtifactBytes: a, MergesBytes: m})` returns a
   `*bekko.Embedder` and reads as "a new bekko embedder". The old names `StaticEmbedder` and
   `NewStaticEmbedder` made the reader ask "static compared to what?". The package name now
   says which model it is, so the type only has to say what it is.
3. **Complexity ledger.**
   ```
   Concepts the developer must learn   +0 / −0   (StaticEmbedder → bekko.Embedder)
   Files they must touch to do X       +0 / −0
   Lines at the call site              +0 / −0   (one import, one constructor name)
   Ways to do the same thing           +1 / −1   (+1 until embed v0.4.0 deletes its copy; L2Normalize → vector.Normalize)
   ```
4. **Where it belongs.** Everything specific to `bekko-embedding-v1-a8m` goes here: special
   token ids 2/1/0, the Metaspace pre-tokenizer, the weight names, the Matryoshka truncation to
   128 dimensions. `embed` keeps only the contract.
5. **What it deletes.** `L2Normalize`: it duplicates `vector.Normalize` from
   `webtyp.com/vector`, which this package already depends on through the encoder.

## Stage 1 — the implementation

Copy from `webtyp/embed` (`main` branch, public: `https://github.com/webtyp/embed`) into this
repository, **package `bekko`**:

| From `embed` | To `bekko` | Changes |
|---|---|---|
| `static_embedder.go` | `embedder.go` | `StaticEmbedder` → `Embedder`, `NewStaticEmbedder` → `New`, error prefix `embed:` → `bekko:`, imports `webtyp.com/transformer` → `webtyp.com/encoder` (package `encoder`), delete `L2Normalize` and call `vector.Normalize(out)` in its place, assertion `var _ embed.Embedder = (*Embedder)(nil)` |
| `dequant.go`, `dequant_test.go` | same names | package clause, error prefix |
| `tokenizer.go` | same name | package clause |
| `weights_bekko.go` | `weights.go` | package clause, `transformer.` → `encoder.` |
| `static_embedder_test.go` | `embedder_test.go` | package `bekko_test`, new names, `TestL2Normalize` → `TestEmbed_OutputIsUnitLength` (every vector from `Embed` with the mock-free path is unit length ±1e-6, skipped without the artifact like the reference test) |
| `count_internal_test.go` | same name | package clause |
| `testdata/reference_vectors.json` | same path | none |

`go.mod`: `go get webtyp.com/embed@v0.3.0 webtyp.com/encoder@v0.2.0 webtyp.com/tokenizer@v0.2.0 webtyp.com/weights@v0.1.0 webtyp.com/vector@v0.1.1 webtyp.com/context@v0.0.23 webtyp.com/fmt@v1.0.0`.

If `vector.Normalize` does not exist with signature `func Normalize(v []float32)`, stop and
report it. Do not keep `L2Normalize`.

## Stage 2 — docs

`README.md`: what the package is (the `bekko-embedding-v1-a8m` embedder for webtyp, runs in
the browser), how to build it (`bekko.New(bekko.Config{...})`), where the artifact comes from
(`webtyp/weightsc` converts the model's `model.safetensors`), and a link to `AGENTS.md`.
Remove the `STATUS` note.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | the files above, `go.mod` | `grep -rn "L2Normalize\|StaticEmbedder\|webtyp.com/transformer" --include=*.go .` → empty; `gotest` and `gotest -tinygo` pass |
| 2 | `README.md` | no `STATUS` line |
