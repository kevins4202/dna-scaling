All five models completed verified 3,051-step research training and the same-prompt frozen-CNN distance evaluation on GPU 7. Exact measured counts are 5,413,780; 10,331,804; 24,886,444; 51,927,104; and 97,352,024. See [the complete results](reports/distance-2048-100m-v1/RESULTS.md) and [training readiness](TRAINING_READY.md).

# Small StripedHyena 2 model family

All five Savanna configurations are created in configs/models/, each with a separate 100-step smoke version. They train from scratch. Verified smoke and final research checkpoints for all five models are stored on Allegro; all five distance evaluations are complete.

| Target label | Layers | Width | Attention heads | Head dimension | Dense projection parameters only |
|---|---:|---:|---:|---:|---:|
| 5M | 24 | 160 | 5 | 32 | 4,915,200 |
| 10M | 24 | 224 | 7 | 32 | 9,633,792 |
| 25M | 24 | 352 | 11 | 32 | 23,789,568 |
| 50M | 24 | 512 | 16 | 32 | 50,331,648 |
| 100M | 24 | 704 | 22 | 32 | 95,158,272 |

The last column is a LOWER BOUND, not an instantiated total. Savanna uses a 3d-by-d input projection and d-by-d output projection in each sequence mixer and each short-Hyena MLP, giving 24 * 8 * d^2 projection weights. Embeddings (512-token character vocabulary), convolution weights, filter parameters, norms and biases add to this. The labels are approximate targets; count actual parameters on the GPU and adjust widths if necessary before research runs.

All sizes retain the same upstream ordering: 6 Hyena-SE, 6 Hyena-MR, 10 Hyena-LI, and 2 attention layers. Short/medium/MLP Hyena group counts scale with width, preserving group dimension 16. Long Hyena uses one group per channel. Both mixer and MLP expansion factors are explicitly 1. Context length is 4096, FP8 is disabled, and precision is BF16. Head dimension is consistently 32; kernel compatibility still requires a GPU smoke run.

## Shared experiment controls

Same tokenizer, source data paths, seed 6200, operator schedule, optimizer, learning rate, warmup, sequence length, and effective batch. These experiments vary width with fixed depth. Separate save/log directories prevent different sizes overwriting each other.

The training configurations assume exactly one GPU, batch size 1, and 8 accumulation steps: 32,768 tokens per optimizer step. 3,051 steps gives 99,975,168 processed tokens for EVERY model, the nearest whole-step budget below 100M. The JSONL dataset contains 100M source characters plus preprocessing separator overhead, so a training run is governed by processed-token budget rather than exactly one pass through the file.

## GPU commands

After installing Savanna and preprocessing the chosen dataset:

```bash
bash scripts/run_model.sh 5m inspect
bash scripts/run_model.sh 5m smoke
bash scripts/run_model.sh 5m train
```

Replace 5m with 10m, 25m, 50m, or 100m. These scripts restrict activity to physical GPU 7; queues wait for availability. Inspection instantiates the actual upstream model on one GPU, prints its parameter count, and saves checkpoints/<size>-parameters.json. It does not train or save random model weights. Smoke configurations have separate checkpoint folders. Training resumes according to Savanna's behavior; inspect existing checkpoint folders before intentionally restarting an experiment.

Use the tiny dataset only for inspect/smoke integration. Full train commands use the verified data/tokenized-100m files via configs/data100m.yml and retain the readiness guard. Do not interpret a tiny-data training run as the planned scaling experiment.

## Local verification

```bash
python3 scripts/test_models.py
```

Checks the full five-model family for identical controls, correct operator counts, valid head/group divisibility, equal token budgets and separate smoke checkpoint paths. Actual counts, 100-step smoke/reload tests, research checkpoints and distance evaluations are verified for all five sizes.

Regenerate configurations with `python3 scripts/build_models.py`; edits to generated model files will be overwritten. The generator uses configs/smoke.yml as the upstream-derived base.
