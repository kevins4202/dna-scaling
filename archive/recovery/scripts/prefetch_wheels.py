import hashlib,json,pathlib,re,time,urllib.request
out=pathlib.Path('/home/ksong37/cis6200/wheels'); out.mkdir(exist_ok=True)
def get(url): return json.load(urllib.request.urlopen(url,timeout=90))
meta=get('https://pypi.org/pypi/torch/2.6.0/json')
packages=[('torch','2.6.0')]
for req in meta['info']['requires_dist']:
 m=re.match(r'(nvidia-[\w-]+)==([\w.]+)',req)
 if m: packages.append(m.groups())
for name,version in packages:
 data=get(f'https://pypi.org/pypi/{name}/{version}/json')
 files=[f for f in data['urls'] if 'x86_64' in f['filename'] and ('manylinux' in f['filename'] or 'linux_' in f['filename']) and ('cp312' in f['filename'] or 'py3-' in f['filename'])]
 if not files: raise RuntimeError(name)
 f=files[0]; target=out/f['filename']; part=target.with_suffix('.part')
 if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==f['digests']['sha256']: continue
 offset=part.stat().st_size if part.exists() else 0
 while offset<f['size']:
  end=min(offset+8*1024*1024,f['size'])-1
  for retry in range(8):
   try:
    req=urllib.request.Request(f['url'],headers={'Range':f'bytes={offset}-{end}'})
    with urllib.request.urlopen(req,timeout=90) as r:
     if r.status!=206: raise RuntimeError('Server did not honor byte range')
     block=r.read()
    if len(block)!=end-offset+1: raise RuntimeError('Incomplete range')
    with part.open('ab') as handle:handle.write(block)
    offset=end+1
    print(name,offset,f['size'],flush=True)
    break
   except Exception:
    if retry==7: raise
    time.sleep(2**min(retry,4))
 if hashlib.sha256(part.read_bytes()).hexdigest()!=f['digests']['sha256']: raise RuntimeError('SHA256 mismatch: '+name)
 part.rename(target)
print('PREFETCH_COMPLETE',flush=True)
