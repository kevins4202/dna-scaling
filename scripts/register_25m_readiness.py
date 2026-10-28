"""Extend existing dataset readiness only after the actual 25M integration test."""
import collections,hashlib,json,math,pathlib,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=ROOT/'reports/training-readiness.json';report=json.loads(path.read_text())
    subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),'5m'],check=True)
    if '25m' in report['models']:
        subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),'25m'],check=True);return
    c=json.loads((ROOT/'configs/models/25m.yml').read_text());baseline=json.loads((ROOT/'configs/models/5m.yml').read_text())
    allowed={'hidden_size','num_attention_heads','num_groups_hyena','num_groups_hyena_medium','num_groups_hyena_short','num_groups_hyena_mlp','save','log-dir'}
    assert {k:v for k,v in c.items() if k not in allowed}=={k:v for k,v in baseline.items() if k not in allowed}
    assert c['train-iters']*c['seq_length']*c['gradient_accumulation_steps']*c['train_micro_batch_size_per_gpu']==99975168
    parameters=json.loads((ROOT/'checkpoints/25m-parameters.json').read_text());assert parameters['total_parameters']==24886444
    verification=json.loads((ROOT/'reports/research/smoke-25m-verification.json').read_text())
    assert verification['checkpoint_training_steps']==100 and verification['strict_state_reload'] and verification['finite_logits'] and verification['new_tokens']==32
    assert verification['total_parameters']==24886444
    log=(ROOT/'logs/25m-project/smoke.log').read_text();assert re.search(r'iteration\s+100/\s+100',log)
    for value in re.findall(r'lm_loss:\s*([\w.+-]+)',log):assert math.isfinite(float(value))
    assert all(int(x)==0 for x in re.findall(r'number of nan iterations:\s*(\d+)',log))
    count=collections.Counter()
    for operators,repeat in c['operator-config']:
        for name in operators:count[name]+=repeat
    assert count=={'hyena_se':6,'hyena_mr':6,'hyena':10,'flash_v2':2}
    report['models']['25m']={'parameters':24886444,'layers':24,'width':352,'attention_heads':11,'head_dimension':32,'context':4096,'operator_counts':dict(count),'tiny_data_100_step_test_passed':True,'strict_smoke_checkpoint_reload_and_generation':True}
    for name in ('configs/models/25m.yml','configs/models/25m_smoke.yml','checkpoints/25m-parameters.json','reports/research/smoke-25m-verification.json'):
        report['required_file_hashes'][name]=sha(ROOT/name)
    backup=ROOT/'provenance/training-readiness-before-25m.json'
    if not backup.exists():backup.write_text(path.read_text())
    report['scope']='Controlled subset experiments; verified readiness extended to 25M after its smoke integration test'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    (ROOT/'reports/25m-parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/check_training_ready.py'),'25m'],check=True)
if __name__=='__main__':main()
