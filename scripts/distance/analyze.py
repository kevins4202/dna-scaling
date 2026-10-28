"""Paired source-group bootstrap of the unchanged frozen CNN metrics."""
import collections, hashlib, json, pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
DATA=ROOT/'data/distance-2048-v1'; OUT=ROOT/'reports/distance-2048-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def estimates(rows,samples,weights):
    output=np.zeros((len(samples),5));total=sum(weights[k] for k in {r['category'] for r in rows})
    for category in sorted({r['category'] for r in rows}):
        part=[r for r in rows if r['category']==category]
        counts=np.array([[s[r['group_id']] for r in part] for s in samples],dtype=float)
        for i in range(5):
            a=np.array([r['natural_logit'] if i==4 else r['natural_chunk_logits'][i] for r in part])
            b=np.array([r['generated_logit'] if i==4 else r['generated_chunk_logits'][i] for r in part])
            delta=b[:,None]-a[None,:];matrix=(delta>0).astype(float)+.5*(delta==0)
            output[:,i]+=weights[category]/total*np.einsum('bi,ij,bj->b',counts,matrix,counts,optimize=True)/counts.sum(axis=1)**2
    return output
def ci(v):return np.quantile(v,[.025,.975]).tolist()
def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,expected in manifest['inputs_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Registered input changed: '+name)
    reports={s:json.loads((OUT/f'{s}-scores.json').read_text()) for s in ('5m','10m')}
    rows=reports['5m']['predictions'];identity=lambda r:(r['id'],r['group_id'],r['category'],r['natural_chunk_logits'])
    if [identity(r) for r in rows]!=[identity(r) for r in reports['10m']['predictions']]:raise ValueError('Mismatched pairs')
    if len(rows)!=100 or any(len(r['natural_chunk_logits'])!=4 or len(r['generated_chunk_logits'])!=4 for report in reports.values() for r in report['predictions']):raise ValueError('Wrong dimensions')
    groups=collections.defaultdict(set)
    for r in rows:groups[r['category']].add(r['group_id'])
    rng=np.random.default_rng(6220);samples=[collections.Counter({r['group_id']:1 for r in rows})]
    for _ in range(2000):
        sample=collections.Counter()
        for category in sorted(groups):
            keys=sorted(groups[category]);sample.update(rng.choice(keys,len(keys),replace=True))
        samples.append(sample)
    values={s:estimates(r['predictions'],samples,manifest['fixed_category_weights']) for s,r in reports.items()};models={}
    for size,array in values.items():
        if not np.isfinite(array).all():raise ValueError('Nonfinite analysis')
        expected=[r['mixture_weighted_stratified_auc'] for r in reports[size]['distance_from_prompt']]+[reports[size]['mixture_weighted_stratified_auc']]
        if not np.allclose(array[0],expected,atol=1e-12,rtol=0):raise ValueError('Metric differs from frozen evaluator')
        metadata=json.loads((DATA/f'generation-{size}.json').read_text())
        if metadata['count']!=100 or metadata['output_sha256']!=sha(DATA/f'generated-{size}.jsonl'):raise ValueError('Generation changed')
        models[size]={'parameters':metadata['parameters'],'distance':[{'start_bp':i*512,'end_bp_exclusive':(i+1)*512,'auc':float(array[0,i]),'ci95':ci(array[1:,i])} for i in range(4)],'aggregate':{'auc':float(array[0,4]),'ci95':ci(array[1:,4])},'last_minus_first':{'difference':float(array[0,3]-array[0,0]),'ci95':ci(array[1:,3]-array[1:,0])}}
    delta=values['10m']-values['5m']
    result={'status':'complete','experiment':'distance-2048-v1','manifest_sha256':sha(DATA/'manifest.json'),'pairs_per_model':100,'source_groups':manifest['source_groups'],'cnn_model_sha256':sha(ROOT/'artifacts/cnn-v1/model.pt'),'models':models,'paired_10m_minus_5m':[{'window':'aggregate' if i==4 else [i*512,(i+1)*512],'difference':float(delta[0,i]),'ci95':ci(delta[1:,i])} for i in range(5)],'bootstrap':manifest['analysis'],'category_counts':manifest['category_counts'],'category_groups':manifest['category_groups'],'limitations':manifest['limitations'],'missing_categories':manifest['missing_categories']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for size,color in [('5m','#2364AA'),('10m','#D05A22')]:
        points=models[size]['distance'];y=[r['auc'] for r in points];bounds=np.array([r['ci95'] for r in points]);x=np.arange(4)*512+256
        ax.plot(x,y,'o-',color=color,label=size.upper());ax.fill_between(x,bounds[:,0],bounds[:,1],alpha=.15,color=color)
    ax.axhline(.5,color='gray',linestyle='--',label='Chance');ax.legend()
    ax.set(xlabel='Distance into continuation (512-base windows)',ylabel='Frozen CNN detection AUROC',ylim=(0,1),xticks=[256,768,1280,1792],xticklabels=['0–512','512–1024','1024–1536','1536–2048'])
    ax.set_title(f'100 matched prompts, {manifest["source_groups"]} source groups')
    fig.text(.5,.015,'Source-group bootstrap 95% intervals; CNN has limited validated sensitivity.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'distance-curve.png',dpi=180);plt.close(fig)
    result['evidence_sha256']={p.name:sha(p) for p in sorted(OUT.glob('*-scores.json'))}
    (OUT/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Frozen CNN 2,048-base distance experiment','','| Bases | 5M AUROC (95% CI) | 10M AUROC (95% CI) |','|---|---|---|']
    for i in range(4):
        cells=[f'{i*512}–{(i+1)*512}']
        for size in ('5m','10m'):
            r=models[size]['distance'][i];cells.append(f'{r["auc"]:.3f} ({r["ci95"][0]:.3f}–{r["ci95"][1]:.3f})')
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','Higher AUROC means easier detection. Aggregate uses mean logits. Intervals are descriptive and unadjusted.','']+['- '+x for x in manifest['limitations']]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
