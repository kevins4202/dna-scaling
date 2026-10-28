import collections,hashlib,json,pathlib,random,re
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'data/mixture-100m-bounded';OUT=ROOT/'data/cnn-pilot'

def candidates(split):
 groups=collections.defaultdict(list)
 for row in map(json.loads,(DATA/f'{split}.jsonl').open()):
  text=row['text'].upper()
  for start in range(0,len(text)-1280+1,1280):
   context=text[start:start+1280]
   if set(context)<=set('ACGT'):
    key=(row['source_shard'],row['source_row'])
    groups[key].append({'category':row['category'],'source_shard':key[0],'source_row':key[1],'source_offset':row.get('source_offset',0)+start,'prefix':context[:256],'natural':context[256:768],'null_natural':context[768:1280]})
 return groups

def main():
 OUT.mkdir(exist_ok=True,parents=True);rng=random.Random(6201);valid=candidates('valid');test=candidates('test')
 keys=list(valid);rng.shuffle(keys)
 if len(keys)<80 or len(test)<20:raise ValueError(f'Insufficient independent source rows: {len(keys)}, {len(test)}')
 selected=[]
 for fold,chosen in [('train',keys[:60]),('calibration',keys[60:80]),('test',rng.sample(list(test),20))]:
  for key in chosen:
   row=dict(rng.choice((test if fold=='test' else valid)[key]));row['fold']=fold
   row['group_id']=hashlib.sha256((row['source_shard']+':'+str(row['source_row'])).encode()).hexdigest()[:16]
   row['id']=len(selected);selected.append(row)
 sets={f:{r['group_id'] for r in selected if r['fold']==f} for f in ('train','calibration','test')}
 assert not (sets['train']&sets['calibration'] or sets['train']&sets['test'] or sets['calibration']&sets['test'])
 for a in ('prefix','natural','null_natural'):
  assert all(len(r[a])==(256 if a=='prefix' else 512) for r in selected)
 (OUT/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
 report={'status':'complete','seed':6201,'prompt_length':256,'continuation_length':512,'natural_control_length':512,'counts':dict(collections.Counter(r['fold'] for r in selected)),'categories_by_fold':{f:dict(collections.Counter(r['category'] for r in selected if r['fold']==f)) for f in sets},'unique_source_groups':{f:len(v) for f,v in sets.items()},'limitations':['Pilot of 100 source groups, with 20 final test groups; confidence intervals will be wide.','Only source rows with 1280 contiguous ACGT bases are eligible, so this does not represent the complete training mixture.','Source-row grouping does not establish species or genome-level independence.','Classifier training and calibration use upstream LM validation data; final classifier test uses upstream LM test data.']}
 (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
