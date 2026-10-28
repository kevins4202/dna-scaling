"""Bind the new checkpoint to the unchanged registered distance experiment."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2];DATA=ROOT/'data/distance-2048-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    path=DATA/'extension-25m.json';extension=json.loads(path.read_text())
    assert extension['model']=='25m' and extension['parameters']==24886444
    for name,expected in extension['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    return extension

def main():
    if (DATA/'extension-25m.json').exists():verify();print('25M extension verified');return
    assert sha(DATA/'manifest.json')=='d8f4ecb84250a303a1dff429d14c1cf4b4684a8afacf8c5bfa877625a434d5a1'
    complete=json.loads((ROOT/'reports/research/25m-complete.json').read_text());assert complete['status']=='verified_complete' and complete['training_steps']==3051 and complete['processed_training_tokens']==99975168
    assert complete['checkpoint_reload_and_generation']['total_parameters']==24886444
    paths=['data/distance-2048-v1/manifest.json','data/distance-2048-v1/prompts.jsonl','data/distance-2048-v1/natural.jsonl','artifacts/cnn-v1/manifest.json','artifacts/cnn-v1/model.pt','checkpoints/25m/global_step3051/mp_rank_00_model_states.pt','configs/models/25m.yml','reports/research/25m-complete.json','reports/distance-2048-v1/complete.json','reports/distance-2048-v1/5m-scores.json','reports/distance-2048-v1/10m-scores.json']
    extension={'version':'distance-2048-25m-v1','model':'25m','parameters':24886444,'training_steps':3051,'processed_training_tokens':99975168,'baseline_experiment':'distance-2048-v1','pairs':100,'prefix_bases':256,'continuation_bases':2048,'batch_size':20,'sampling':'Unchanged original manifest settings and per-prompt seeds','analysis':'Same 2000 category-stratified source-group resamples, seed 6220, paired across 5M/10M/25M and all distances; descriptive unadjusted intervals','contrasts':['25m minus 5m','25m minus 10m','last minus first window for 25m'],'scope':'Exploratory extension authorized after observing the first two sizes; CNN and reference stay fixed','inputs_sha256':{p:sha(ROOT/p) for p in paths}}
    (DATA/'extension-25m.json').write_text(json.dumps(extension,indent=2)+'\n');verify();print('Registered unchanged-prompt 25M evaluation extension')
if __name__=='__main__':main()
