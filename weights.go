package bekko

import (
	"webtyp.com/encoder"
	"webtyp.com/fmt"
	"webtyp.com/weights"
)

func bekkoA8mConfig() encoder.Config {
	return encoder.Config{
		NumLayers:          4,
		Heads:              6,
		Dim:                384, // NATIVE — no confundir con los 128 de salida (D0)
		FFNDim:             1152,
		GlobalEveryNLayers: 3,
		LocalWindow:        128,
		GlobalRopeTheta:    160000.0,
		LocalRopeTheta:     160000.0,
		Eps:                1e-5,
		Pooling:            encoder.PoolingMean,
		Activation:         encoder.ActivationGELU, // config.json: "hidden_activation": "gelu"
	}
}

func loadLayer(get func(string) (weights.Tensor, error), i int) (encoder.LayerWeights, error) {
	prefix := fmt.Sprintf("layers.%d.", i)

	wqkv, err := get(fmt.Sprintf("%sattn.Wqkv.weight", prefix))
	if err != nil {
		return encoder.LayerWeights{}, err
	}
	wqkvF, err := dequantFull(wqkv)
	if err != nil {
		return encoder.LayerWeights{}, err
	}

	wo, err := get(fmt.Sprintf("%sattn.Wo.weight", prefix))
	if err != nil {
		return encoder.LayerWeights{}, err
	}
	woF, err := dequantFull(wo)
	if err != nil {
		return encoder.LayerWeights{}, err
	}

	wi, err := get(fmt.Sprintf("%smlp.Wi.weight", prefix))
	if err != nil {
		return encoder.LayerWeights{}, err
	}
	wiF, err := dequantFull(wi)
	if err != nil {
		return encoder.LayerWeights{}, err
	}

	mlpWo, err := get(fmt.Sprintf("%smlp.Wo.weight", prefix))
	if err != nil {
		return encoder.LayerWeights{}, err
	}
	mlpWoF, err := dequantFull(mlpWo)
	if err != nil {
		return encoder.LayerWeights{}, err
	}

	mlpNorm, err := get(fmt.Sprintf("%smlp_norm.weight", prefix))
	if err != nil {
		return encoder.LayerWeights{}, err
	}
	mlpNormF, err := dequantFull(mlpNorm)
	if err != nil {
		return encoder.LayerWeights{}, err
	}

	var attnNormF []float32
	attnNorm, err := get(fmt.Sprintf("%sattn_norm.weight", prefix))
	if err == nil {
		attnNormF, err = dequantFull(attnNorm)
		if err != nil {
			return encoder.LayerWeights{}, err
		}
	}

	return encoder.LayerWeights{
		WqkvT:         wqkvF,
		WoT:           woF,
		WiT:           wiF,
		MlpWoT:        mlpWoF,
		AttnNormGamma: attnNormF,
		MlpNormGamma:  mlpNormF,
	}, nil
}

// loadWeights dequantizes every layer tensor EXCEPT the token embedding table (kept lazy —
// see dequantFull's doc comment) into a encoder.Weights ready for Encode. artifact.Tensor
// panics on no code path here — every name below is checked, a missing one is a hard error,
// never a zero-valued layer silently fed into Encode.
func loadWeights(art *weights.Artifact, cfg encoder.Config) (encoder.Weights, error) {
	get := func(name string) (weights.Tensor, error) {
		t, ok := art.Tensor(name)
		if !ok {
			return weights.Tensor{}, fmt.Err("bekko: artifact missing tensor ", name)
		}
		return t, nil
	}

	embedNorm, err := get("embeddings.norm.weight")
	if err != nil {
		return encoder.Weights{}, err
	}
	embedNormF, err := dequantFull(embedNorm)
	if err != nil {
		return encoder.Weights{}, err
	}

	finalNorm, err := get("final_norm.weight")
	if err != nil {
		return encoder.Weights{}, err
	}
	finalNormF, err := dequantFull(finalNorm)
	if err != nil {
		return encoder.Weights{}, err
	}

	w := encoder.Weights{EmbedNormGamma: embedNormF, FinalNormGamma: finalNormF}
	for i := 0; i < cfg.NumLayers; i++ {
		lw, err := loadLayer(get, i)
		if err != nil {
			return encoder.Weights{}, err
		}
		w.Layers = append(w.Layers, lw)
	}
	return w, nil
}
