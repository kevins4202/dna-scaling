"""Pin, download and verify official released weights; preserve resumable parts."""
import hashlib,json,pathlib,re,shutil,time,os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi,hf_hub_download
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'checkpoints/released-evo2-7b-base';REPORT=ROOT/'reports/evo7b-baseline'
REPO='arcinstitute/evo2_7b_base'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);path=REPORT/'download-manifest.json'
    if path.exists():manifest=json.loads(path.read_text())
    else:
        info=HfApi().model_info(REPO,files_metadata=True);files=[]
        for item in info.siblings:
            name=item.rfilename
            if name=='config.json' or name=='evo2_7b_base.pt' or re.fullmatch(r'evo2_7b_base\.pt\.part\d+',name):
                lfs=item.lfs;expected=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
                files.append({'name':name,'size':item.size,'expected_sha256':expected})
        assert any(f['name'].startswith('evo2_7b_base.pt') for f in files)
        manifest={'repo':REPO,'revision':info.sha,'files':files,'status':'pinned'};path.write_text(json.dumps(manifest,indent=2)+'\n')
    for item in manifest['files']:
        destination=OUT/'source'/item['name']
        if not destination.exists() or destination.stat().st_size!=item['size']:
            print('DOWNLOADING',item['name'],item['size'],flush=True)
            hf_hub_download(repo_id=REPO,filename=item['name'],revision=manifest['revision'],local_dir=str(OUT/'source'))
        assert destination.stat().st_size==item['size']
        actual=sha(destination)
        if item['expected_sha256']:assert actual==item['expected_sha256'],item['name']
        item['sha256']=actual;print('VERIFIED_FILE',item['name'],actual,flush=True)
    direct=OUT/'source/evo2_7b_base.pt'
    if direct.exists():checkpoint=direct
    else:
        parts=sorted((OUT/'source').glob('evo2_7b_base.pt.part*'),key=lambda p:int(p.name.split('.part')[-1]));assert parts
        assert [int(p.name.split('.part')[-1]) for p in parts]==list(range(len(parts)))
        checkpoint=OUT/'evo2_7b_base.pt'
        if not checkpoint.exists() or checkpoint.stat().st_size!=sum(p.stat().st_size for p in parts):
            temporary=checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as output:
                for part in parts:
                    print('MERGING',part.name,flush=True)
                    with part.open('rb') as source:shutil.copyfileobj(source,output,8*1024*1024)
            temporary.replace(checkpoint)
    manifest.update(status='verified',checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=sha(checkpoint));path.write_text(json.dumps(manifest,indent=2)+'\n');print('RELEASED_CHECKPOINT_VERIFIED',manifest['revision'],manifest['checkpoint_bytes'],flush=True)
if __name__=='__main__':main()
