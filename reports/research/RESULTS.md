# Completed 5M and 10M research runs

Both runs completed sequentially on Allegro GPU 7. Each trained for 3,051 optimizer steps and 99,975,168 processed tokens on the same prepared subset. Final checkpoint hashes, strict reload, finite logits, and 32-base A/C/G/T-constrained generation were verified.

| Model | Parameters | Validation loss | Test loss | Test perplexity |
|---|---:|---:|---:|---:|
| 5m | 5,413,780 | 1.240848 | 1.187362 | 3.278420 |
| 10m | 10,331,804 | 1.241422 | 1.187922 | 3.280258 |

The loss difference is very small. These single runs do not establish that either model size is better. Biological generation quality has not yet been evaluated. The subset and taxonomy limitations in TRAINING_READY.md still apply.

Checkpoints remain on Allegro under /home/ksong37/cis6200/checkpoints/{5m,10m}/global_step3051/. Compact training histories and completion/verification reports are saved alongside this file.

Both runs used TORCH_COMPILE_DISABLE=1 after the initial launch exposed a CUDA-graph error with gradient accumulation. Other architecture, optimizer, data and token-budget settings were unchanged. GPU 7 was idle after completion.
