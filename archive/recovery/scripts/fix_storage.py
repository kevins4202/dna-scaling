from pathlib import Path
root=Path('/home/ksong37/cis6200')
(root/'tmp').mkdir(exist_ok=True)
p=root/'scripts/setup_allegro.sh'
s=p.read_text()
s=s.replace('export MAX_JOBS=4', 'export TMPDIR=/home/ksong37/cis6200/tmp\nexport PIP_NO_CACHE_DIR=1\nexport MAX_JOBS=4')
s=s.replace('python -m pip install torch==2.6.0', 'python -m pip install --no-deps /home/ksong37/cis6200/wheels/*.whl\npython -m pip install torch==2.6.0')
p.write_text(s)
q=root/'scripts/queue_gpu7_smoke.sh'
s=q.read_text().replace('export MAX_JOBS=4','export TMPDIR=/home/ksong37/cis6200/tmp\nexport PIP_NO_CACHE_DIR=1\nexport MAX_JOBS=4')
q.write_text(s)
