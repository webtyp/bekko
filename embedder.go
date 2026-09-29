package bekko

import (
	"webtyp.com/embed"
	"webtyp.com/vector"

	"webtyp.com/context"
	"webtyp.com/encoder"
	"webtyp.com/fmt"
	"webtyp.com/tokenizer"
	"webtyp.com/weights"
)

// truncatedDim is 128, not 384 or 64 — Matryoshka (D0). Verified against the real model
// card (huggingface.co/hotchpotch/bekko-embedding-v1-a8m, "Truncation and Quantization"):
// 128 dims loses -7.05% quality ("memory-constrained indexes"), the smallest size the model
// card does not explicitly discourage for retrieval — 64 dims loses -17.51% and is labeled
// "not for quality-sensitive retrieval" by the model's own authors.
const truncatedDim = 128

// Config assembles a Embedder. Both byte slices are already fetched — this package
// never imports webtyp.com/fetch or decides where the artifact lives (D5's IndexedDB
// caching, the download itself: the caller's job, not this adapter's).
type Config struct {
	ArtifactBytes []byte // the .wtypw file
	MergesBytes   []byte // the companion .merges file, same run of weightsc
}

type Embedder struct {
	id     string
	art    *weights.Artifact
	tokEmb weights.Tensor // token embedding table — NEVER dequantized in full, see dequantFull
	cfg    encoder.Config
	w      encoder.Weights
	bpe    *tokenizer.BPE
}

func New(cfg Config) (*Embedder, error) {
	art, err := weights.Open(cfg.ArtifactBytes)
	if err != nil {
		return nil, fmt.Err("bekko: opening artifact: ", err)
	}
	tokEmb, ok := art.Tensor("embeddings.tok_embeddings.weight")
	if !ok {
		return nil, fmt.Err("bekko: artifact missing embeddings.tok_embeddings.weight")
	}

	tcfg := bekkoA8mConfig()
	w, err := loadWeights(art, tcfg)
	if err != nil {
		return nil, err
	}
	bpe, err := loadTokenizer(art, cfg.MergesBytes)
	if err != nil {
		return nil, err
	}

	return &Embedder{
		id:     fmt.Sprintf("%s/d%d", art.ID, truncatedDim),
		art:    art,
		tokEmb: tokEmb,
		cfg:    tcfg,
		w:      w,
		bpe:    bpe,
	}, nil
}

func (e *Embedder) Dim() int { return truncatedDim }

func (e *Embedder) ID() string { return e.id }

// CountTokens is the length of the token sequence Embed feeds the encoder for text.
func (e *Embedder) CountTokens(text string) int {
	return len(e.bpe.Encode(nil, text))
}

func (e *Embedder) Close() error { return nil }

func (e *Embedder) Embed(ctx *context.Context, texts []string, dst []float32) error {
	expected := len(texts) * truncatedDim
	if len(dst) != expected {
		return fmt.Err("bekko: dst length mismatch, expected ", expected, " got ", len(dst))
	}
	if len(texts) == 0 {
		return nil
	}

	for i, text := range texts {
		ids := e.bpe.Encode(nil, text)
		seqLen := len(ids)
		if seqLen == 0 {
			return fmt.Err("bekko: empty token sequence for text at index ", i)
		}

		tokenEmbeds := make([]float32, seqLen*e.cfg.Dim)
		for pos, id := range ids {
			if int(id) < 0 || int(id) >= e.tokEmb.Shape[0] {
				return fmt.Err("bekko: token id out of vocab range: ", id)
			}
			dequantRow(e.tokEmb, int(id), tokenEmbeds[pos*e.cfg.Dim:(pos+1)*e.cfg.Dim])
		}

		pooled, err := encoder.Encode(e.cfg, e.w, tokenEmbeds, seqLen)
		if err != nil {
			return fmt.Err("bekko: Encode: ", err)
		}

		out := dst[i*truncatedDim : (i+1)*truncatedDim]
		copy(out, pooled[:truncatedDim])
		// A Matryoshka truncation is only a valid embedding AFTER renormalizing: the first
		// 128 components of a unit 384-vector are not themselves unit length.
		vector.Normalize(out)
	}
	return nil
}

var _ embed.Embedder = (*Embedder)(nil)
