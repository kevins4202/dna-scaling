"""Matched released reference versus the unchanged fixed-budget model family."""
import collections,sys
from common import ROOT,OUT,ORIGINAL,SCORES,guard,read,write,sha
sys.path.insert(0,str(ROOT/'scripts/distance'))
from analyze import estimates,ci
import numpy as np

def main():
    protocol=guard();manifest=read(ORIGINAL/'manifest.json')
    reports={s:read(ROOT/p) for s,p in SCORES.items()};reports['released_7b']=read(OUT/'released-7b-scores.json')
    reference=reports['5m']['predictions']
    identity=lambda r:(str(r['id']),r['group_id'],r['category'],r['natural_chunk_logits'],r['natural_logit'])
    for report in reports.values():
        assert [identity(r) for r in report['predictions']]==[identity(r) for r in reference],'Unmatched scoring records'
    groups=collections.defaultdict(set)
    for r in reference:groups[r['category']].add(r['group_id'])
    rng=np.random.default_rng(6220);samples=[collections.Counter({r['group_id']:1 for r in reference})]
    for _ in range(2000):
        sample=collections.Counter()
        for category in sorted(groups):
            keys=sorted(groups[category]);sample.update(rng.choice(keys,len(keys),replace=True))
        samples.append(sample)
    arrays={s:estimates(r['predictions'],samples,manifest['fixed_category_weights']) for s,r in reports.items()}
    values={}
    for s,a in arrays.items():
        assert np.isfinite(a).all()
        expected=[x['mixture_weighted_stratified_auc'] for x in reports[s]['distance_from_prompt']]+[reports[s]['mixture_weighted_stratified_auc']]
        assert np.allclose(a[0],expected,atol=1e-12,rtol=0)
        values[s]={'windows':[{'auc':float(a[0,i]),'ci95':ci(a[1:,i])} for i in range(4)],'aggregate':{'auc':float(a[0,4]),'ci95':ci(a[1:,4])},'last_minus_first':{'difference':float(a[0,3]-a[0,0]),'ci95':ci(a[1:,3]-a[1:,0])}}
    contrasts={s:[{'window':i if i<4 else 'aggregate','difference':float((arrays['released_7b']-a)[0,i]),'ci95':ci((arrays['released_7b']-a)[1:,i])} for i in range(5)] for s,a in arrays.items() if s!='released_7b'}
    categories={}
    for cat in sorted(groups):
        categories[cat]={'source_groups':len(groups[cat]),'models':{}}
        for s,r in reports.items():
            a=estimates([x for x in r['predictions'] if x['category']==cat],samples,{cat:1})
            categories[cat]['models'][s]={'aggregate_auc':float(a[0,4]),'ci95':ci(a[1:,4])}
    result={'status':'complete','protocol_sha256':sha(OUT/'protocol.json'),'pairs':100,'source_groups':len({r['group_id'] for r in reference}),'models':values,'released_minus_small_model':contrasts,'categories':categories,'analysis':protocol['analysis'],'limitations':protocol['limitations'],'evidence_sha256':{'reports/evo7b-baseline/released-7b-scores.json':sha(OUT/'released-7b-scores.json'),**{p:sha(ROOT/p) for p in SCORES.values()}}}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for s,v in values.items():
        y=np.array([x['auc'] for x in v['windows']]);bounds=np.array([x['ci95'] for x in v['windows']])
        line=ax.plot(np.arange(4)*512+256,y,'o-' if s=='released_7b' else '.--',linewidth=2.7 if s=='released_7b' else 1,alpha=1 if s=='released_7b' else .65,label='Released 7B (external reference)' if s=='released_7b' else s.upper())[0]
        if s=='released_7b':ax.fill_between(np.arange(4)*512+256,bounds[:,0],bounds[:,1],color=line.get_color(),alpha=.15)
    ax.axhline(.5,color='gray',linestyle=':');ax.set(xlabel='Distance into continuation (bases)',ylabel='Frozen CNN detection AUROC',ylim=(0,1),xticks=[256,768,1280,1792],xticklabels=['0–512','512–1024','1024–1536','1536–2048']);ax.legend(fontsize=8)
    ax.set_title('Same 100 prompts; released model has a different training history')
    fig.tight_layout();fig.savefig(OUT/'distance-curve.png',dpi=180);plt.close(fig)
    lines=['# Released Evo 2 7B reference','','Higher AUROC means generated DNA is easier for the frozen CNN to detect.','','| Model | Aggregate AUROC | 95% interval |','|---|---:|---|']
    for s,v in values.items():
        a=v['aggregate'];lines.append(f'| {s} | {a["auc"]:.4f} | {a["ci95"][0]:.4f} to {a["ci95"][1]:.4f} |')
    lines+=['','## Category results','','| Category | Groups | Released 7B AUROC | 95% interval |','|---|---:|---:|---|']
    for cat,v in categories.items():
        a=v['models']['released_7b'];lines.append(f'| {cat} | {v["source_groups"]} | {a["aggregate_auc"]:.4f} | {a["ci95"][0]:.4f} to {a["ci95"][1]:.4f} |')
    lines+=['','## Interpretation limits','']+['- '+x for x in protocol['limitations']]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(OUT/'complete.json',result);print('ANALYSIS_COMPLETE',values['released_7b'],flush=True)
if __name__=='__main__':main()
