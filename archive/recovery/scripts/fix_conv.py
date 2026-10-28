from pathlib import Path
import sys
root=Path(sys.argv[1])
for p in (root/'configs').rglob('*.yml'):
 s=p.read_text().replace('"fast_conv_proj": true','"fast_conv_proj": false')
 p.write_text(s)
