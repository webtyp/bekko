# bekko
<img src="docs/img/badges.svg">

The `bekko-embedding-v1-a8m` embedding model for webtyp. It implements `embed.Embedder` (text in,
128-dimension vectors out, plus `CountTokens`) in Go, running in the browser under TinyGo. It
composes `webtyp/tokenizer`, `webtyp/weights` and `webtyp/encoder`, and implements none of them
itself. Formerly `embed.StaticEmbedder`.

## Getting started

```go
emb, err := bekko.New(bekko.Config{ArtifactBytes: wtypw, MergesBytes: merges})
var e embed.Embedder = emb // use it wherever an embed.Embedder is asked for
```

The two byte slices are the output of `webtyp/weightsc` run on the model's original files
(`model.safetensors`, `config.json`, `tokenizer.json` from `hotchpotch/bekko-embedding-v1-a8m`):

```bash
weightsc -in <model dir> -out bekko-embedding-v1-a8m.wtypw \
         -merges-out bekko-embedding-v1-a8m.merges -id bekko-embedding-v1-a8m -version 1
```

This package never downloads or caches the artifact. The application reads it (in the browser,
from `webtyp/opfs`) and passes the bytes.

**Tests against the real model.** Put (or symlink) the two files in `testdata/`. Git ignores
them, and without them the reference tests skip. They verify cosine ≥ 0.999 against the model's
own output, unit-length vectors, and that `CountTokens` equals what `Embed` reads.

## Documentation

- [Agent guide](AGENTS.md): rules for anyone changing this library.
- [Last executed plan](docs/LAST_PLAN_EXECUTED.md): the move out of `webtyp/embed`.
- Why this model and why 128 dimensions:
  [semantic-search master plan](https://github.com/webtyp/retrieval/blob/main/docs/SEMANTIC_SEARCH_MASTER_PLAN.md) (D0, D5).
