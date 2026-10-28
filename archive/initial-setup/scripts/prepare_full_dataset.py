import concurrent.futures
import hashlib
import json
import pathlib
import random
import re
import sys
import time
import urllib.request
import urllib.parse
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sample_opengenome as sampler

ROOT = pathlib.Path('/home/ksong37/cis6200')
OUT = ROOT / 'data/mixture-100m'
CACHE = ROOT / 'data/source-shards'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
ORIGINAL_REQUEST = sampler.request


def download(shard):
    path = CACHE / pathlib.Path(shard['path']).name
    if path.exists() and path.stat().st_size == shard['size']:
        return path
    part = path.with_suffix(path.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    url = f'https://huggingface.co/datasets/{sampler.REPO}/resolve/{sampler.REVISION}/' + urllib.parse.quote(shard['path'])
    while offset < shard['size']:
        end = min(offset + 8*1024*1024, shard['size']) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(url + f'?download=true&offset={offset}', headers={'Range': f'bytes={offset}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    content_range = response.headers.get('Content-Range', '')
                    if response.status != 206 or not content_range.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Byte range not honored: ' + str(response.status) + ' ' + content_range)
                    block = response.read()
                if len(block) != end-offset+1:
                    raise RuntimeError('Incomplete range')
                with part.open('ab') as f: f.write(block)
                offset=end+1
                if offset % (256*1024*1024) == 0 or offset == shard['size']:
                    print(f'DOWNLOAD {path.name}: {offset}/{shard["size"]}', flush=True)
                break
            except Exception as error:
                if attempt == 7: raise
                print('RETRY',path.name,offset,str(error),flush=True)
                time.sleep(min(30,2**attempt))
    part.rename(path)
    return path


def cached_request(url):
    if '/resolve/' in url:
        name=pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        return (CACHE/name).open('rb')
    return ORIGINAL_REQUEST(url)


def run():
    manifest_path=OUT/'manifest.json'
    manifest={'status':'incomplete','dataset':sampler.REPO,'revision':sampler.REVISION,'mode':'full-random-shard-subset',
              'seed':6200,'raw_weights':sampler.WEIGHTS,'raw_weight_sum':sum(sampler.WEIGHTS.values()),
              'weight_source':sampler.SOURCE,'token_definition':'ASCII source characters, excluding appended EOD tokens',
              'sampling':'One uniformly shuffled complete shard per category and upstream split, then seeded record-priority sampling to exact normalized category character quotas. Approximate within-category distribution; no prefix sampling.',
              'limitations':['Random shard subset, not a scan of all OpenGenome2.','Species balance and homology separation are unverified.','Variable record sizes and final cropping affect sampling.'], 'splits':{}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    args=SimpleNamespace(seed=6200,mode='full',shards_per_category=1,smoke_scan_chars=32768)
    jobs=[]
    for split in ('train','valid','test'):
        for category in sampler.WEIGHTS:
            files=sampler.list_files(category,split)
            random.Random(f'6200/{category}/{split}').shuffle(files)
            jobs.append((category,split,files[0]))
    (OUT/'source-selection.json').write_text(json.dumps(jobs,indent=2)+'\n')
    total_bytes=sum(x[2]['size'] for x in jobs)
    print('PLANNED_COMPRESSED_BYTES',total_bytes,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for path in pool.map(download,[x[2] for x in jobs]):
            print('READY',path.name,flush=True)
    sampler.request=cached_request
    saved=OUT/'category-samples';saved.mkdir(exist_ok=True)
    for split in ('train','valid','test'):
        budget=100_000_000 if split=='train' else 1_000_000
        targets=sampler.quotas(budget)
        output=[];detail={}
        for category in sampler.WEIGHTS:
            cache=saved/f'{split}-{category}.json'
            if cache.exists():
                result=json.loads(cache.read_text());records,report=result['records'],result['report']
            else:
                print('SAMPLING',split,category,targets[category],flush=True)
                records,report=sampler.sample_category(category,split,targets[category],args)
                cache.write_text(json.dumps({'records':records,'report':report})+'\n')
            output.extend(records);detail[category]=report
        random.Random(f'6200/{split}/shuffle').shuffle(output)
        path=OUT/f'{split}.jsonl'
        with path.open('w') as f:
            for record in output:f.write(json.dumps(record)+'\n')
        manifest['splits'][split]={'characters':sum(len(x['text']) for x in output),'records':len(output),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'categories':detail}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['status']='complete'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('DATASET_COMPLETE',flush=True)

if __name__=='__main__':
    run()
