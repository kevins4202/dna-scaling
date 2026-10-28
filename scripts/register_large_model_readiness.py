"""Extend hash-guarded readiness after a measured 50M/100M smoke test."""
import argparse,collections,hashlib,json,math,pathlib,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('size',choices=('50m','100m'));size=parser.parse_args().size
    path=ROOT/'reports/training-readiness.json';report=json.loads(path.read_text())
    subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),'5m'],check=True)
    if size in report['models']:
        subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),size],check=True);return
    c=json.loads((ROOT/f'configs/models/{size}.yml').read_text());base=json.loads((ROOT/'configs/models/5m.yml').read_text())
    allowed={'hidden_size','num_attention_heads','num_groups_hyena','num_groups_hyena_medium','num_groups_hyena_short','num_groups_hyena_mlp','save','log-dir'}
    assert {k:v for k,v in c.items() if k not in allowed}=={k:v for k,v in base.items() if k not in allowed}
    assert c['train-iters']*c['seq_length']*c['gradient_accumulation_steps']*c['train_micro_batch_size_per_gpu']==99975168
    params=json.loads((ROOT/f'checkpoints/{size}-parameters.json').read_text())
    assert params['hidden_size']==c['hidden_size'] and params['num_layers']==24 and params['padded_vocab_size']==512
    assert params['total_parameters']=={'50m':51927104,'100m':97352024}[size]
    assert params['trainable_parameters']==params['total_parameters']>24*8*c['hidden_size']**2
    verification=json.loads((ROOT/f'reports/research/smoke-{size}-verification.json').read_text())
    assert verification['checkpoint_training_steps']==100 and verification['strict_state_reload'] and verification['finite_logits'] and verification['new_tokens']==32
    assert verification['total_parameters']==params['total_parameters']
    log=(ROOT/f'logs/large-models/{size}-smoke.log').read_text();assert re.search(r'iteration\s+100/\s+100',log)
    losses=re.findall(r'lm_loss:\s*([\w.+-]+)',log);assert losses and all(math.isfinite(float(x)) for x in losses)
    assert all(int(x)==0 for x in re.findall(r'number of nan iterations:\s*(\d+)',log))
    operators=collections.Counter()
    for names,repeat in c['operator-config']:
        for name in names:operators[name]+=repeat
    assert operators=={'hyena_se':6,'hyena_mr':6,'hyena':10,'flash_v2':2}
    backup=ROOT/f'provenance/training-readiness-before-{size}.json'
    if not backup.exists():backup.write_text(path.read_text())
    report['models'][size]={'parameters':params['total_parameters'],'layers':24,'width':c['hidden_size'],'attention_heads':c['num_attention_heads'],'head_dimension':32,'context':4096,'operator_counts':dict(operators),'tiny_data_100_step_test_passed':True,'strict_smoke_checkpoint_reload_and_generation':True}
    for name in (f'configs/models/{size}.yml',f'configs/models/{size}_smoke.yml',f'checkpoints/{size}-parameters.json',f'reports/research/smoke-{size}-verification.json'):
        report['required_file_hashes'][name]=sha(ROOT/name)
    report['scope']='Controlled subset experiments; measured model readiness added only after smoke and strict reload tests'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    (ROOT/f'reports/{size}-parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),size],check=True)
if __name__=='__main__':main()
