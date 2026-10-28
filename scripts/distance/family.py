"""Register larger checkpoints against the immutable shared distance experiment."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2];DATA=ROOT/'data/distance-2048-v1'
SIZES=('5m','10m','25m','50m','100m')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def report_dir(size):
    assert size in SIZES
    return ROOT/('reports/distance-2048-v1' if size in ('5m','10m') else f'reports/distance-2048-{size}-v1')
def verify(size):
    assert size in ('50m','100m');extension=json.loads((DATA/f'extension-{size}.json').read_text())
    assert extension['parameters']=={'50m':51927104,'100m':97352024}[size]
    assert extension['model']==size and extension['evaluation_models']==list(SIZES[:SIZES.index(size)+1])
    for name,expected in extension['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    return extension

def register(size):
    assert size in ('50m','100m')
    if (DATA/f'extension-{size}.json').exists():verify(size);print(size,'extension verified');return
    assert sha(DATA/'manifest.json')=='d8f4ecb84250a303a1dff429d14c1cf4b4684a8afacf8c5bfa877625a434d5a1'
    complete=json.loads((ROOT/f'reports/research/{size}-complete.json').read_text());params=json.loads((ROOT/f'checkpoints/{size}-parameters.json').read_text())
    assert complete['status']=='verified_complete' and complete['training_steps']==3051 and complete['processed_training_tokens']==99975168
    assert complete['checkpoint_reload_and_generation']['total_parameters']==params['total_parameters']
    paths=['data/distance-2048-v1/manifest.json','data/distance-2048-v1/prompts.jsonl','data/distance-2048-v1/natural.jsonl','artifacts/cnn-v1/manifest.json','artifacts/cnn-v1/model.pt',f'checkpoints/{size}/global_step3051/mp_rank_00_model_states.pt',f'checkpoints/{size}-parameters.json',f'configs/models/{size}.yml',f'reports/research/{size}-complete.json']
    previous=SIZES[:SIZES.index(size)]
    for old in previous:
        paths.append(str((report_dir(old)/f'{old}-scores.json').relative_to(ROOT)))
        paths.append(f'data/distance-2048-v1/generation-{old}.json')
        paths.append(f'data/distance-2048-v1/generated-{old}.jsonl')
    extension={'version':f'distance-2048-{size}-v1','model':size,'parameters':params['total_parameters'],'evaluation_models':list(previous)+[size],'training_steps':3051,'processed_training_tokens':99975168,'pairs':100,'prefix_bases':256,'continuation_bases':2048,'batch_size':20,'sampling':'Original registered prompts, natural references, per-prompt seeds and sampling settings unchanged','analysis':'Same 2000 category-stratified whole-source-group bootstrap resamples, seed 6220, paired across models and distances; descriptive unadjusted intervals; pairwise size contrasts and last-minus-first for each size','scope':'Exploratory family extension authorized after smaller-model results; no CNN retraining, no reference changes and no automatic significance/scaling-law claim','inputs_sha256':{p:sha(ROOT/p) for p in paths}}
    (DATA/f'extension-{size}.json').write_text(json.dumps(extension,indent=2)+'\n');verify(size);print('Registered',size,'unchanged-prompt distance extension')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('size',choices=('50m','100m'));register(parser.parse_args().size)
