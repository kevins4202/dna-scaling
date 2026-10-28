"""Attribute existing source-group uncertainty without changing frozen CNN scores."""
import collections,importlib.util,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reports/category-uncertainty-v1'
SIZES=('5m','10m','25m','50m','100m')
def score_path(size):return ROOT/('reports/distance-2048-v1' if size in ('5m','10m') else f'reports/distance-2048-{size}-v1')/f'{size}-scores.json'
def values(rows,counts,window):
    real=np.array([r['natural_logit'] if window==4 else r['natural_chunk_logits'][window] for r in rows]);fake=np.array([r['generated_logit'] if window==4 else r['generated_chunk_logits'][window] for r in rows])
    difference=fake[:,None]-real[None,:];matrix=(difference>0).astype(float)+.5*(difference==0)
    return np.einsum('bi,ij,bj->b',counts,matrix,counts,optimize=True)/counts.sum(axis=1)**2

def main():
    reports={size:json.loads(score_path(size).read_text()) for size in SIZES};manifest=json.loads((ROOT/'data/distance-2048-v1/manifest.json').read_text());weights=manifest['fixed_category_weights'];categories=sorted(reports['5m']['per_category_auc']);total=sum(weights[k] for k in categories)
    output={'status':'complete','scope':'Exploratory diagnosis of the existing100-prompt evaluation; no CNN or input changes','bootstrap':'10000 independent category-stratified whole-source-group resamples, seed6226; category weighted-variance shares and paired difference variance shares','categories':{},'models':{},'paired_to_5m':{}}
    rng=np.random.default_rng(6226);boot={size:{} for size in SIZES}
    for category in categories:
        part={size:[r for r in reports[size]['predictions'] if r['category']==category] for size in SIZES};groups=sorted({r['group_id'] for r in part['5m']});n=len(groups)
        multiplicities=rng.multinomial(n,np.full(n,1/n),size=10000);indices=[groups.index(r['group_id']) for r in part['5m']];counts=np.vstack((np.ones(len(indices)),multiplicities[:,indices]))
        output['categories'][category]={'pairs':len(indices),'source_groups':n,'normalized_weight':weights[category]/total,'models':{}}
        for size in SIZES:
            array=np.stack([values(part[size],counts,window) for window in range(5)],axis=1);boot[size][category]=array
            output['categories'][category]['models'][size]={'aggregate_auc':float(array[0,4]),'aggregate_ci95':np.quantile(array[1:,4],[.025,.975]).tolist(),'distance_auc':array[0,:4].tolist(),'distance_ci95':np.quantile(array[1:,:4],[.025,.975],axis=0).T.tolist()}
    for size in SIZES:
        weighted={c:weights[c]/total*boot[size][c][1:,4] for c in categories};variances={c:float(np.var(v,ddof=1)) for c,v in weighted.items()};sumvar=sum(variances.values())
        contributions={c:v/sumvar for c,v in variances.items()}
        full=sum(weighted.values());without_euk=sum(v for c,v in weighted.items() if c!='eukaryotic_genic_windows')/(1-weights['eukaryotic_genic_windows']/total)
        output['models'][size]={'category_variance_share':contributions,'bootstrap_ci95':np.quantile(full,[.025,.975]).tolist(),'variance_sum':sumvar,'sensitivity_without_euk_renormalized_ci95':np.quantile(without_euk,[.025,.975]).tolist(),'note':'Removing a category changes the estimand; this is a diagnostic, not a replacement result'}
        if size!='5m':
            differences={c:weights[c]/total*(boot[size][c][1:,4]-boot['5m'][c][1:,4]) for c in categories};v={c:float(np.var(x,ddof=1)) for c,x in differences.items()};t=sum(v.values())
            output['paired_to_5m'][size]={'category_difference_variance_share':{c:x/t for c,x in v.items()},'aggregate_difference_ci95':np.quantile(sum(differences.values()),[.025,.975]).tolist()}
    euk=[r for r in reports['5m']['predictions'] if r['category']=='eukaryotic_genic_windows'];influence=[]
    for group in sorted({r['group_id'] for r in euk}):
        record={'group_id':group,'removed_pairs':sum(r['group_id']==group for r in euk),'models':{}}
        for size in SIZES:
            rows=[r for r in reports[size]['predictions'] if r['category']=='eukaryotic_genic_windows' and r['group_id']!=group];auc=float(values(rows,np.ones((1,len(rows))),4)[0]);old=reports[size]['per_category_auc']['eukaryotic_genic_windows']
            record['models'][size]={'euk_auc_without_group':auc,'weighted_total_shift':weights['eukaryotic_genic_windows']/total*(auc-old)}
        influence.append(record)
    output['euk_leave_one_group_out']=influence
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Existing category-level uncertainty diagnosis','','| Category | Pairs | Source groups | Weight | 5M AUC | 10M AUC | 25M AUC | 50M AUC | 100M AUC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c,r in output['categories'].items():lines.append('| '+c+' | '+str(r['pairs'])+' | '+str(r['source_groups'])+f' | {r["normalized_weight"]:.1%} | '+' | '.join(f'{r["models"][s]["aggregate_auc"]:.3f}' for s in SIZES)+' |')
    lines+=['','## Eukaryotic contribution to uncertainty','']
    for size in SIZES:lines.append(f'- {size.upper()}: {output["models"][size]["category_variance_share"]["eukaryotic_genic_windows"]:.1%} of the sum of weighted category bootstrap variances.')
    lines+=['','Variance shares describe the current group-bootstrap uncertainty, not biological causes. Categories with very few groups can have unreliable intervals, including spuriously zero estimated variance. Leave-one-group-out shifts and all per-category intervals are retained in diagnosis.json. CNN sensitivity and generation/training-seed variation are separate limitations.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
