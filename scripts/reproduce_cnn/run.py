import json
import os
import subprocess
import sys
from pathlib import Path

from prepare import ROOT, OUT, digest, write


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, sha in protocol['inputs_sha256'].items():
        assert digest(ROOT / name) == sha, name
    import numpy, pandas, sklearn, torch
    write(OUT / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
          'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__,
          'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
          'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
          'PYTHONHASHSEED': os.environ['PYTHONHASHSEED']})
    for index, task in enumerate(protocol['tasks']):
        status = ROOT / 'logs/cnn-original-v1/status'
        status.write_text(f'TRAINING_FOLD {index + 1}/{len(protocol["tasks"])} {json.dumps(task)}\n')
        log = ROOT / 'logs/cnn-original-v1' / f'fold-{index:03d}.log'
        with log.open('a') as handle:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/reproduce_cnn/run_fold.py'), str(index)],
                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print('Completed fold', index, task, flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_cnn/report.py')], cwd=ROOT, check=True)
    for name, sha in protocol['inputs_sha256'].items():
        assert digest(ROOT / name) == sha, name
    after = {name: digest(ROOT / name) for name in protocol['frozen_cnn_sha256']}
    assert after == protocol['frozen_cnn_sha256']
    write(OUT / 'cnn-integrity-after.json', after)
    write(OUT / 'complete.json', {'status': 'verified_complete', 'folds': len(protocol['tasks']),
          'protocol_sha256': digest(OUT / 'protocol.json'), 'frozen_cnn_unchanged': True,
          'summary_sha256': digest(OUT / 'summary.json')})
    (ROOT / 'logs/cnn-original-v1/status').write_text('ORIGINAL_CNN_REPRODUCTION_VERIFIED_COMPLETE\n')


if __name__ == '__main__':
    main()
