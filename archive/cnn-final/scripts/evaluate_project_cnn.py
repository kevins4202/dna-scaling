import argparse
import collections
import hashlib
import json
import pathlib

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    from score_project_cnn import default_release, load_release, score_text
except ImportError:
    from score import default_release, load_release, score_text

CATEGORY_WEIGHTS = {'gtdb_v220_imgpr':18, 'metagenomes':24, 'imgvr_untagged':3,
    'mrna_splice_promoter':9, 'mrna':9, 'eukaryotic_genic_windows':35, 'organelle':.5, 'ncrna':2, 'promoters':.02}


def auc(real, generated):
    return float(roc_auc_score([0]*len(real)+[1]*len(generated), list(real)+list(generated)))


def metrics(rows):
    categories = collections.defaultdict(list)
    for row in rows:
        categories[row['category']].append(row)
    per_category = {name:auc([r['natural_logit'] for r in values], [r['generated_logit'] for r in values])
                    for name,values in categories.items()}
    total = sum(CATEGORY_WEIGHTS[k] for k in categories)
    return {'mixture_weighted_stratified_auc':float(sum(CATEGORY_WEIGHTS[k]*v for k,v in per_category.items())/total),
            'pooled_auc':auc([r['natural_logit'] for r in rows], [r['generated_logit'] for r in rows]),
            'per_category_auc':per_category, 'represented_reference_weight':total,
            'missing_categories':sorted(set(CATEGORY_WEIGHTS)-set(categories))}


def bootstrap(rows, seed=6213, count=2000):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    # Resample complete source groups within category, preserving fixed category weights.
    by_category = collections.defaultdict(list)
    for name in names:
        values = groups[name]
        if len({r['category'] for r in values})!=1:
            raise ValueError('Source group crosses categories')
        by_category[values[0]['category']].append(name)
    estimates = []
    for _ in range(count):
        selected = [r for keys in by_category.values() for name in rng.choice(keys,len(keys),replace=True) for r in groups[name]]
        estimates.append(metrics(selected)['mixture_weighted_stratified_auc'])
    return np.quantile(estimates,[.025,.975]).tolist()


def read(path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(r['id']) for r in rows]
    if len(set(ids))!=len(ids) or not ids:
        raise ValueError('Each input needs nonempty, unique matching ids')
    return dict(zip(ids,rows))


def main():
    parser=argparse.ArgumentParser(description='Evaluate matched continuations with fixed CNN v1, weighted to the eligible OpenGenome2 mixture.')
    parser.add_argument('--natural',type=pathlib.Path,required=True)
    parser.add_argument('--generated',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--release',type=pathlib.Path,default=default_release())
    args=parser.parse_args();torch.set_num_threads(4)
    real,fake=read(args.natural),read(args.generated)
    if real.keys()!=fake.keys():raise ValueError('Natural and generated ids must match exactly')
    model,manifest=load_release(args.release,args.device);predictions=[]
    if hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()!=manifest['files_sha256']['evaluate.py']:
        raise ValueError('Evaluation implementation differs from the frozen release')
    for id in sorted(real):
        natural=real[id].get('text',real[id].get('natural'))
        generated=fake[id].get('generated',fake[id].get('text'))
        if not isinstance(natural,str) or not isinstance(generated,str) or len(natural)!=len(generated):
            raise ValueError(f'Mismatched continuation lengths: {id}')
        category=real[id].get('category')
        if category not in CATEGORY_WEIGHTS or 'group_id' not in real[id]:
            raise ValueError('Natural records need an eligible category and actual source group_id')
        a,ac=score_text(model,natural,args.device);b,bc=score_text(model,generated,args.device)
        predictions.append({'id':id,'group_id':str(real[id]['group_id']),'category':category,
            'natural_logit':a,'generated_logit':b,'natural_chunk_logits':ac,'generated_chunk_logits':bc})
    if len({len(r['natural_chunk_logits']) for r in predictions})!=1:
        raise ValueError('All evaluated continuations must use the same length')
    result={'cnn_version':manifest['version'],'model_sha256':manifest['files_sha256']['model.pt'],
        'validation_status':manifest.get('validation_status','unspecified'),
        'natural_input_sha256':hashlib.sha256(args.natural.read_bytes()).hexdigest(),
        'generated_input_sha256':hashlib.sha256(args.generated.read_bytes()).hexdigest(),
        'pairs':len(predictions),'source_groups':len({r['group_id'] for r in predictions}),
        'continuation_length':512*len(predictions[0]['natural_chunk_logits']),**metrics(predictions),
        'source_group_bootstrap_ci95':bootstrap(predictions),
        'distance_from_prompt':[{'start_bp':i*512,'end_bp_exclusive':(i+1)*512,
            **metrics([dict(r,natural_logit=r['natural_chunk_logits'][i],generated_logit=r['generated_chunk_logits'][i]) for r in predictions])}
            for i in range(len(predictions[0]['natural_chunk_logits']))],
        'interpretation':'Higher AUROC means easier detection; chance is 0.5. Primary AUROC averages category-specific AUROCs with fixed mixture weights, renormalized over represented categories. Group-bootstrap intervals do not prove species independence.',
        'predictions':predictions}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('EVALUATED',result['pairs'],'pairs; weighted AUROC',result['mixture_weighted_stratified_auc'])


if __name__=='__main__':main()
