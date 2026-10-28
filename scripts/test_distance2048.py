"""Test the paired AUROC implementation against direct replicated observations."""
import collections,importlib.util,pathlib,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('distance_analysis',ROOT/'scripts/distance/analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Tests(unittest.TestCase):
    def test_group_multiplicity_and_ties(self):
        rows=[]
        for category,group,a,b in [('a','g1',1,1),('a','g1',2,3),('a','g2',0,2),('b','g3',5,4)]:
            rows.append(dict(category=category,group_id=group,natural_logit=a,generated_logit=b,natural_chunk_logits=[a]*4,generated_chunk_logits=[b]*4))
        samples=[collections.Counter(g1=2,g2=1,g3=1),collections.Counter(g1=0,g2=3,g3=1)]
        actual=module.estimates(rows,samples,{'a':3,'b':1})
        for n,sample in enumerate(samples):
            expected=0
            for category,weight in [('a',.75),('b',.25)]:
                repeated=[r for r in rows if r['category']==category for _ in range(sample[r['group_id']])]
                pairs=[float(b['generated_logit']>a['natural_logit'])+.5*(b['generated_logit']==a['natural_logit']) for a in repeated for b in repeated]
                expected+=weight*sum(pairs)/len(pairs)
            np.testing.assert_allclose(actual[n],expected,atol=1e-15)
if __name__=='__main__':unittest.main()
