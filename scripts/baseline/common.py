"""Registered released-model reference, separate from the fixed-budget family."""
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/evo7b-baseline';DATA=ROOT/'data/evo7b-baseline-v1';ORIGINAL=ROOT/'data/distance-2048-v1'
CNN='33621e4a90aafbb6c925ff52f7b6c45d74301a66f902cc561a05b83bc3da14b5'
CNN_MANIFEST='73d1c7823755cb722430c937918e67160284d62de64683cc14646dda4cca3f59'
MANIFEST='d8f4ecb84250a303a1dff429d14c1cf4b4684a8afacf8c5bfa877625a434d5a1'
SCORES={'5m':'reports/distance-2048-v1/5m-scores.json','10m':'reports/distance-2048-v1/10m-scores.json',**{s:f'reports/distance-2048-{s}-v1/{s}-scores.json' for s in ('25m','50m','100m')}}
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def rows(p):return [json.loads(s) for s in pathlib.Path(p).read_text().splitlines() if s.strip()]
def guard():
    assert sha(ORIGINAL/'manifest.json')==MANIFEST,'Original manifest changed'
    assert sha(ROOT/'artifacts/cnn-v1/model.pt')==CNN,'Frozen CNN changed'
    assert sha(ROOT/'artifacts/cnn-v1/manifest.json')==CNN_MANIFEST,'Frozen CNN manifest changed'
    protocol=read(OUT/'protocol.json')
    for name,digest in protocol['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    return protocol

def register():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    release=read(OUT/'download-manifest.json');assert release['status']=='verified'
    assert sha(ROOT/release['checkpoint'])==release['checkpoint_sha256']
    names=['data/distance-2048-v1/'+n for n in ('manifest.json','prompts.jsonl','natural.jsonl')]+list(SCORES.values())+['reports/evo7b-baseline/download-manifest.json']
    names+=['scripts/baseline/'+n for n in ('common.py','generate.py','analyze.py','validate.py','compat.py')]
    vendors={n:subprocess.check_output(['git','-C',str(ROOT/'vendor'/n),'rev-parse','HEAD'],text=True).strip() for n in ('evo2','vortex')}
    protocol={'experiment':'evo7b-baseline-v1','model':'evo2_7b_base','checkpoint':release,'vendor_commits':vendors,'inputs_sha256':{n:sha(ROOT/n) for n in names},'cnn_sha256':CNN,'cnn_manifest_sha256':CNN_MANIFEST,'sampling':{'prefix_bases':256,'generated_bases':2048,'allowed':'ACGT','temperature':1.0,'top_k':None,'top_p':None,'seeds':'621900 + original integer id','batch_size':1,'bos':False},'inference':{'precision':'official BF16 fallback, FP32 long-filter parameters','transformer_engine':False,'optional_kernels':False,'cached':False,'mode':'full_prefix','cache_rejection':'Original recurrent cache exceeded prespecified probability L1 .03; retained failed evidence and use full-prefix reference path without changing thresholds or sampling','attention_compatibility':'process-local FlashAttention 2.6.3 forward tuple adapter; kernels unchanged','probe_bases':32,'probe_max_logit_absolute_difference':0.25,'probe_max_conditional_probability_l1':0.03},'analysis':{'resamples':2000,'seed':6220,'unit':'category-stratified whole source group','paired':True},'limitations':['External pretrained reference, not a controlled fixed-100M-token size comparison.','Overlap of released-model pretraining with evaluation sources is unknown.','Original 100 prompts from 90 groups; only six eukaryotic groups.','Frozen CNN has limited validated sensitivity; AUROC is detectability, not biological function.','Pairwise intervals are exploratory, unadjusted for multiple comparisons.']}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'Registered protocol differs'
    else:write(OUT/'protocol.json',protocol)
    guard();print('PROTOCOL_REGISTERED',sha(OUT/'protocol.json'),flush=True)
if __name__=='__main__':register()
