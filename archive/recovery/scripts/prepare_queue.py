from pathlib import Path
p = Path('/home/ksong37/cis6200/scripts/setup_allegro.sh')
s = p.read_text().replace('--index-url https://download.pytorch.org/whl/cu124', '--index-url https://pypi.org/simple')
s = s.replace("python -c 'import torch;", "cd /home/ksong37/cis6200/vendor/savanna\npython -c 'import torch;")
p.write_text(s)
