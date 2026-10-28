"""Cumulative paired comparison using the original frozen-CNN AUROC estimator."""
import argparse,collections,json
import numpy as np
from analyze import estimates,ci
from family import DATA,ROOT,SIZES,report_dir,sha,verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('size',choices=('50m','100m'));size=parser.parse_args().size
    extension=verify(size);manifest=json.loads((DATA/'manifest.json').read_text());sizes=extension['evaluation_models'];out=report_dir(size)
    reports={s:json.loads((report_dir(s)/f'{s}-scores.json').read_text()) for s in sizes}
    rows=reports['5m']['predictions'];identity=lambda r:(r['id'],r['group_id'],r['category'],r['natural_chunk_logits'])
    assert len(rows)==100
    for report in reports.values():
        assert [identity(r) for r in report['predictions']]==[identity(r) for r in rows]
        assert report['continuation_length']==2048
    groups=collections.defaultdict(set)
    for row in rows:groups[row['category']].add(row['group_id'])
    rng=np.random.default_rng(6220);samples=[collections.Counter({r['group_id']:1 for r in rows})]
    for _ in range(2000):
        sample=collections.Counter()
        for category in sorted(groups):
            keys=sorted(groups[category]);sample.update(rng.choice(keys,len(keys),replace=True))
        samples.append(sample)
    values={s:estimates(reports[s]['predictions'],samples,manifest['fixed_category_weights']) for s in sizes};models={}
    for name,array in values.items():
        assert np.isfinite(array).all()
        expected=[r['mixture_weighted_stratified_auc'] for r in reports[name]['distance_from_prompt']]+[reports[name]['mixture_weighted_stratified_auc']]
        assert np.allclose(array[0],expected,atol=1e-12,rtol=0)
        metadata=json.loads((DATA/f'generation-{name}.json').read_text());assert metadata['output_sha256']==sha(DATA/f'generated-{name}.jsonl')
        models[name]={'parameters':metadata['parameters'],'distance':[{'start_bp':i*512,'end_bp_exclusive':(i+1)*512,'auc':float(array[0,i]),'ci95':ci(array[1:,i])} for i in range(4)],'aggregate':{'auc':float(array[0,4]),'ci95':ci(array[1:,4])},'last_minus_first':{'difference':float(array[0,3]-array[0,0]),'ci95':ci(array[1:,3]-array[1:,0])}}
    contrasts={}
    for i,large in enumerate(sizes):
        for small in sizes[:i]:
            delta=values[large]-values[small]
            contrasts[f'{large}_minus_{small}']=[{'window':'aggregate' if j==4 else [j*512,(j+1)*512],'difference':float(delta[0,j]),'ci95':ci(delta[1:,j])} for j in range(5)]
    result={'status':'complete','experiment':extension['version'],'extension_sha256':sha(DATA/f'extension-{size}.json'),'baseline_manifest_sha256':sha(DATA/'manifest.json'),'pairs_per_model':100,'source_groups':manifest['source_groups'],'models':models,'paired_contrasts':contrasts,'category_counts':manifest['category_counts'],'category_groups':manifest['category_groups'],'limitations':manifest['limitations']+[extension['scope']],'cnn_model_sha256':sha(ROOT/'artifacts/cnn-v1/model.pt'),'bootstrap':extension['analysis'],'evidence_sha256':{str((report_dir(s)/f'{s}-scores.json').relative_to(ROOT)):sha(report_dir(s)/f'{s}-scores.json') for s in sizes}}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name,color in zip(sizes,('#2364AA','#D05A22','#32834F','#7A4BA0','#AC4570')):
        points=models[name]['distance'];bounds=np.array([r['ci95'] for r in points]);x=np.arange(4)*512+256
        ax.plot(x,[r['auc'] for r in points],'o-',color=color,label=name.upper());ax.fill_between(x,bounds[:,0],bounds[:,1],alpha=.08,color=color)
    ax.axhline(.5,color='gray',linestyle='--',label='Chance');ax.legend(fontsize=9)
    ax.set(xlabel='Distance into continuation (512-base windows)',ylabel='Frozen CNN detection AUROC',ylim=(0,1),xticks=[256,768,1280,1792],xticklabels=['0–512','512–1024','1024–1536','1536–2048'])
    ax.set_title('100 matched prompts, 90 source groups')
    fig.text(.5,.015,'Source-group bootstrap 95% intervals; CNN has limited validated sensitivity.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'distance-curve.png',dpi=180);plt.close(fig)
    lines=[f'# Frozen CNN distance experiment through {size.upper()}','','| Continuation bases | '+' | '.join(s.upper()+' AUROC (95% CI)' for s in sizes)+' |','|---|'+'---|'*len(sizes)]
    for i in range(5):
        cells=['Aggregate' if i==4 else f'{i*512}–{(i+1)*512}']
        for name in sizes:
            row=models[name]['aggregate'] if i==4 else models[name]['distance'][i];cells.append(f'{row["auc"]:.3f} ({row["ci95"][0]:.3f}–{row["ci95"][1]:.3f})')
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','Higher AUROC means easier detection. Aggregate averages logits. Intervals are descriptive and unadjusted.','']+['- '+v for v in result['limitations']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');(out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
