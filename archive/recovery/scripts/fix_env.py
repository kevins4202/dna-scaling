from pathlib import Path
import sys
root=Path(sys.argv[1])
for name in ('run_model.sh','preprocess_gpu.sh'):
 p=root/'scripts'/name
 s=p.read_text().replace('source .venv-gpu/bin/activate', 'if [[ -n "${CIS6200_ENV_BIN:-}" ]]; then\n    export PATH="$CIS6200_ENV_BIN:$PATH"\nelse\n    source .venv-gpu/bin/activate\nfi')
 p.write_text(s)
q=root/'scripts/queue_gpu7_smoke.sh'
if q.exists():
 s=q.read_text().replace('export CUDA_VISIBLE_DEVICES=7', 'export CUDA_VISIBLE_DEVICES=7\nexport CIS6200_ENV_BIN=/home/ksong37/miniconda3/envs/cis6200/bin')
 q.write_text(s)
