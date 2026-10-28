"""Fresh, screened, precision-focused held-out DNA; never consult CNN predictions."""
import argparse,collections,concurrent.futures,gzip,hashlib,json,pathlib,random,re,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import sample_opengenome as source
OUT=ROOT/'data/distance-2048-expanded-v1'
QUOTAS={'eukaryotic_genic_windows':160,'metagenomes':96,'gtdb_v220_imgpr':64,'mrna_splice_promoter':48,'mrna':48,'imgvr_untagged':40,'ncrna':32,'organelle':24}
LIMITS={c:20_000_000 for c in QUOTAS};LIMITS.update(gtdb_v220_imgpr=100_000_000,ncrna=50_000_000)
SEED=622600

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with path.open() as handle:yield from (json.loads(line) for line in handle if line.strip())
def key(row):return hashlib.sha256((row['source_shard']+':'+str(row['source_row'])).encode()).hexdigest()[:16]
def identity(category,record,shard,number):
    if category=='gtdb_v220_imgpr' and isinstance(record,str):
        cleaned=record.removeprefix('NZ_');match=re.fullmatch(r'([A-Z]{4,6})(\d{2})\d{6,8}(?:\.\d+)?',cleaned)
        return ('wgs-project/'+''.join(match.groups()),'wgs_project_proxy') if match else ('accession/'+record,'accession')
    return ('row/'+shard+':'+str(number),'source_row')
def priority(value):return hashlib.sha256((str(SEED)+'/'+value).encode()).hexdigest()
def excluded():
    rows=set();biological=set();used=set();maximum={}
    paths=[ROOT/f'data/mixture-100m-bounded/{s}.jsonl' for s in ('train','valid','test')]+list((ROOT/'data/reference-independent').glob('*.jsonl'))
    paths += [ROOT/'data/cnn-pilot/prompts.jsonl',ROOT/'data/cnn-project-final/prompts.jsonl',ROOT/'data/distance-2048-v1/prompts.jsonl']
    for path in paths:
        if not path.exists():continue
        for row in read(path):
            rows.add(key(row));used.add(row['source_shard']);maximum[row['source_shard']]=max(maximum.get(row['source_shard'],0),row['source_row'])
            group,kind=identity(row['category'],row.get('source_record'),row['source_shard'],row['source_row'])
            if kind!='source_row':biological.add(group)
    for fold in ('train','calibration','test'):
        for row in read(ROOT/f'data/cnn-final-mixed/{fold}.jsonl'):
            if row['domain']=='project':rows.add(row['group_id'])
    return rows,biological,used,maximum

def collect_one(category,excluded_rows,excluded_ids,used_shards,maximum):
    path=OUT/f'pool-{category}.jsonl';receipt=OUT/f'pool-{category}.json'
    if receipt.exists():
        report=json.loads(receipt.read_text());assert sha(path)==report['sha256'];print('POOL_VERIFIED',category,report['groups'],flush=True);return report
    inventory=json.loads((ROOT/'reports/category-uncertainty-v1/upstream-test-inventory.json').read_text());assert inventory['revision']==source.REVISION
    files=inventory['categories'][category]['files'];groups={};scanned=eligible=0;provenance=[]
    supplemental={'eukaryotic_genic_windows':(40,4_000_000,50_000_000,24),'gtdb_v220_imgpr':(16,6_000_000,50_000_000,16),'imgvr_untagged':(8,4_000_000,20_000_000,50)}
    if category in supplemental:
        count,soft,hard,minimum=supplemental[category];files=source.list_files(category,'train')
        if category!='eukaryotic_genic_windows':files=[f for f in files if f['path'] not in used_shards]
        files=sorted(files,key=lambda f:priority(f['path']))[:count]
    else:soft=hard=LIMITS[category];minimum=0
    for shard in files:
        assert ('_test_' in shard['path'] or category in supplemental) and '/'+category+'/' in shard['path']
        chars=records=skipped_chars=0;digest=hashlib.sha256();local_groups=set();skip_rows=maximum.get(shard['path'],0);stopped=False
        with source.request(f'https://huggingface.co/datasets/{source.REPO}/resolve/{source.REVISION}/'+shard['path']) as response,gzip.GzipFile(fileobj=response) as stream:
            for number,line in enumerate(stream,1):
                raw=json.loads(line);digest.update(line);text=raw['text'].upper();records=number
                if number<=skip_rows:skipped_chars+=len(text);continue
                chars+=len(text)
                group,kind=identity(category,raw.get('record'),shard['path'],number)
                original={'source_shard':shard['path'],'source_row':number}
                if key(original) not in excluded_rows and group not in excluded_ids:
                    runs=[(m.start(),len(m.group())-2304+1) for m in re.finditer(r'[ACGT]{2304,}',text)]
                    count=sum(n for _,n in runs)
                    if count:
                        rng=random.Random(f'{SEED}/{category}/{shard["path"]}/{number}');selected=rng.randrange(count)
                        for start,n in runs:
                            if selected<n:offset=start+selected;break
                            selected-=n
                        sequence=text[offset:offset+2304];rank=priority(shard['path']+':'+str(number));eligible+=1;local_groups.add(group)
                        candidate={'category':category,'group_id':hashlib.sha256((category+'/'+group).encode()).hexdigest()[:16],'group_type':kind,'source_identity':group,'source_shard':shard['path'],'source_row':number,'source_offset':offset,'source_record':raw.get('record'),'source_record_length':len(text),'prefix':sequence[:256],'natural':sequence[256:],'selection_rank':rank}
                        if group not in groups or rank<groups[group]['selection_rank']:groups[group]=candidate
                if number%10000==0:print('SCAN',category,number,chars,len(groups),flush=True)
                if chars>=hard or (chars>=soft and len(local_groups)>=minimum):stopped=True;break
        scanned+=chars;provenance.append({'shard':shard['path'],'raw_characters_scanned':chars,'raw_records_scanned':records,'scanned_prefix_sha256':digest.hexdigest(),'skipped_prefix_rows':skip_rows,'skipped_prefix_characters':skipped_chars,'heldout_groups_seen':len(local_groups),'complete_scan':not stopped})
    chosen=sorted(groups.values(),key=lambda r:priority(r['category']+'/'+r['source_identity']))[:12*QUOTAS[category]]
    temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(row)+'\n' for row in chosen));temporary.replace(path)
    report={'category':category,'groups':len(groups),'eligible_records':eligible,'pool_rows':len(chosen),'scanned_characters':scanned,'provenance':provenance,'sha256':sha(path),'group_types':dict(collections.Counter(r['group_type'] for r in chosen)),'collection_plan':{'soft_raw_characters_per_shard':soft,'hard_raw_characters_per_shard':hard,'minimum_groups_per_shard':minimum,'shards':len(files),'source_split':'train' if category in supplemental else 'test'}}
    receipt.write_text(json.dumps(report,indent=2)+'\n');print('POOL_READY',category,len(chosen),'groups_seen',len(groups),flush=True);return report

def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():verify();return
    a,b,used,maximum=excluded()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(collect_one,c,a,b,used,maximum) for c in QUOTAS]
        for job in jobs:job.result()

def aligned(line):
    p=line.split('\t');return int(p[3])-int(p[2])>=256 and int(p[9])/max(1,int(p[10]))>=.90

def screen(candidates,target,label):
    path=OUT/f'{label}.paf'
    with path.open('w') as output:
        subprocess.run(['minimap2','-x','asm10','-c','-t','4','-N','50',str(target),str(OUT/'candidates.fasta')],stdout=output,check=True)
    return {int(line.split('\t')[0][1:]) for line in path.read_text().splitlines() if aligned(line)}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():verify();return
    collect();candidates=[row for category in QUOTAS for row in read(OUT/f'pool-{category}.jsonl')]
    (OUT/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    (OUT/'candidates.fasta').write_text(''.join(f'>q{i}\n{r["prefix"]+r["natural"]}\n' for i,r in enumerate(candidates)))
    with (OUT/'lm-train.fasta').open('w') as handle:
        for i,row in enumerate(read(ROOT/'data/mixture-100m-bounded/train.jsonl')):handle.write(f'>t{i}\n{row["text"].upper()}\n')
    development=set()
    for fold in ('train','calibration','test'):
        for row in read(ROOT/f'data/cnn-final-mixed/{fold}.jsonl'):
            development.update(row[k].upper() for k in ('natural','generated','null_natural') if k in row)
    for split in ('valid','test'):
        development.update(row['text'].upper() for row in read(ROOT/f'data/mixture-100m-bounded/{split}.jsonl'))
    development.update(r['prefix']+r['natural'] for r in read(ROOT/'data/distance-2048-v1/prompts.jsonl'))
    (OUT/'development.fasta').write_text(''.join(f'>d{i}\n{s}\n' for i,s in enumerate(sorted(development))))
    rejected=screen(candidates,OUT/'lm-train.fasta','vs-lm-train')|screen(candidates,OUT/'development.fasta','vs-development')
    selfpaf=OUT/'within-panel.paf'
    with selfpaf.open('w') as output:subprocess.run(['minimap2','-x','asm10','-c','-t','4','-N','50',str(OUT/'candidates.fasta'),str(OUT/'candidates.fasta')],stdout=output,check=True)
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for line in selfpaf.read_text().splitlines():
        p=line.split('\t');a,b=int(p[0][1:]),int(p[5][1:])
        if a!=b and a not in rejected and b not in rejected and aligned(line):parent[find(b)]=find(a)
    components=collections.defaultdict(list)
    for i,row in enumerate(candidates):
        if i not in rejected:components[find(i)].append(i)
    retained=[min(indices,key=lambda i:priority(candidates[i]['category']+'/'+candidates[i]['source_identity'])) for indices in components.values()]
    pools=collections.defaultdict(list)
    for i in retained:pools[candidates[i]['category']].append(candidates[i])
    available={c:len(pools[c]) for c in QUOTAS};print('SCREENED_AVAILABILITY',available,flush=True)
    (OUT/'screening.json').write_text(json.dumps({'candidates':len(candidates),'development_rejected':len(rejected),'within_panel_components':len(components),'available':available},indent=2)+'\n')
    assert all(available[c]>=n for c,n in QUOTAS.items()),'Insufficient screened distinct groups; extend bounded source collection before registering'
    selected=[];chunks=set()
    for category,n in QUOTAS.items():
        pool=sorted(pools[category],key=lambda r:priority(r['category']+'/'+r['source_identity']))
        for row in pool[:n]:
            row=dict(row,id=len(selected),seed=SEED+len(selected));row.pop('selection_rank');selected.append(row)
            for j in range(4):
                sequence=row['natural'][j*512:(j+1)*512];reverse=sequence.translate(str.maketrans('ACGT','TGCA'))[::-1];value=min(sequence,reverse)
                assert value not in chunks,'Duplicate natural chunks';chunks.add(value)
    assert len(selected)==512 and len({r['group_id'] for r in selected})==512
    (OUT/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected));(OUT/'natural.jsonl').write_text(''.join(json.dumps(dict(r,text=r['natural']))+'\n' for r in selected))
    protected=['data/distance-2048-expanded-v1/prompts.jsonl','data/distance-2048-expanded-v1/natural.jsonl','data/distance-2048-v1/manifest.json','artifacts/cnn-v1/manifest.json','artifacts/cnn-v1/model.pt']
    for size in ('5m','10m','25m','50m','100m'):protected.extend([f'checkpoints/{size}/global_step3051/mp_rank_00_model_states.pt',f'configs/models/{size}.yml'])
    baseline=[r for r in read(ROOT/'data/distance-2048-v1/prompts.jsonl')];composition={}
    for category in QUOTAS:
        real=[r['natural'] for r in selected if r['category']==category];old=[r['natural'] for r in baseline if r['category']==category]
        gc=lambda texts:sum(x.count('G')+x.count('C') for x in texts)/sum(map(len,texts)) if texts else None
        composition[category]={'new_gc':gc(real),'old_gc':gc(old),'groups':QUOTAS[category],'group_types':dict(collections.Counter(r['group_type'] for r in selected if r['category']==category))}
    manifest={'version':'distance-2048-expanded-v1','pairs':512,'source_groups':512,'category_counts':QUOTAS,'category_groups':QUOTAS,'fixed_category_weights':source.WEIGHTS,'represented_reference_weight':sum(source.WEIGHTS[c] for c in QUOTAS),'missing_categories':['promoters'],'prefix_bases':256,'continuation_bases':2048,'seed':SEED,'batch_size':20,'models':['5m','10m','25m','50m','100m'],'upstream_revision':source.REVISION,'upstream_split':'Fresh held-out rows from mixed upstream splits; full per-category provenance retained in pool receipts','collection_plans':{c:json.loads((OUT/f'pool-{c}.json').read_text()).get('collection_plan',{'soft_raw_characters_per_shard':LIMITS[c],'hard_raw_characters_per_shard':LIMITS[c],'source_split':'test'}) for c in QUOTAS},'selection':'One2304-base random eligible window per source group, hashed group selection within bounded fresh-source prefixes, skipping all previously observed rows on reused eukaryotic shards; WGS project proxy used for identifiable GTDB accessions, row fallback otherwise. Eukaryotic sources span all40 training shards after previously observed row prefixes; GTDB/virus use previously unseen training shards; other categories use upstream test records. Similarity components contribute at most one record across all categories.','exclusions':'All LM train/valid/test and historical reference/CNN project source rows excluded; known accession/WGS identities excluded. AllCNNtrain/calibration/test natural/generated/null, prior distance natural andLMtrain/valid/test sequences screened by minimap2 asm10 -c at >=90% identity over >=256 query bases.','sampling':'Same ACGT-constrained categorical temperature1,no top-k/top-p or repetition penalty; matched per-prompt seeds across all five models','analysis':{'bootstrap':2000,'seed':6220,'method':'Category-stratified whole-source-group bootstrap; identical resamples for all sizes/distances; fixed original category weights renormalized over eligible categories','contrasts':'All pairwise model differences and last-minus-first for each model; descriptive unadjusted intervals','primary':'Aggregate and four per-distance mixture-weighted within-category AUROCs; report all per-category estimates and intervals','diagnosis':'Report weighted category variance shares and leave-one-source-group-out influence; compare uncertainty widths to prior100-prompt evaluation without pooling the two sets'},'composition':composition,'limitations':['This is a fresh precision-focused panel designed after inspecting old category uncertainty; it is not a uniform token sample of OpenGenome2. Original weights are applied at analysis.','Bounded fresh-source collection and2304 contiguous ACGT eligibility limit population coverage; promoters are ineligible.','Most upstream records lack taxonomy. WGS grouping improves GTDB grouping but512 source groups do not establish512 unrelated species.','Sequence matching reduces detected overlap but does not prove phylogenetic independence.','One generation seed per source and one training seed per size; sampling intervals omit training-seed uncertainty.','Frozen CNN remains a qualified limited-sensitivity diagnostic; no biological-function or automatic scaling-law claim.'],'inputs_sha256':{p:sha(ROOT/p) for p in protected}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');verify();print('REGISTERED_EXPANDED_PANEL',len(selected),flush=True)

def verify():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['pairs']==512 and manifest['source_groups']==512
    for name,expected in manifest['inputs_sha256'].items():assert sha(ROOT/name)==expected,name
    return manifest
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('collect','prepare','verify'));mode=parser.parse_args().mode
    {'collect':collect,'prepare':prepare,'verify':verify}[mode]()
