# LM_Dec_Long_InfiniAttention

InfiniAttention - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: (new) lucidrains/infini-transformer-pytorch 0.2.1
- Upstream / license: lucidrains/infini-transformer-pytorch (MIT), pip 0.2.1

## Differences from the source ([Unified] protocol)

- segments of 128, compressive memory detached every 2 segments

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.16 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.683 -> 0.007 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.612 / test 1.614 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.496 / test 1.505 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 520 MB, inference 42 MB
- Peak GPU memory at length 8192 (batch 1): train 1944 MB, inference 46 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.01 (keys removed 0.01); 4 pairs @ 2048: 0.02 (keys removed 0.01); 16 pairs @ 512: 0.02 (keys removed 0.01); 16 pairs @ 2048: 0.02 (keys removed 0.02)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.02 (keys removed 0.02); 4 pairs @ 2048: 0.02 (keys removed 0.02); 4 pairs @ 8192: 0.02 (keys removed 0.01); 16 pairs @ 512: 0.01 (keys removed 0.02); 16 pairs @ 2048: 0.02 (keys removed 0.02); 16 pairs @ 8192: 0.02 (keys removed 0.01)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy infini-transformer-pytorch==0.2.1
python run.py check infini
python run.py lm02 infini      # enwik8 is downloaded to data/ on first use
```
