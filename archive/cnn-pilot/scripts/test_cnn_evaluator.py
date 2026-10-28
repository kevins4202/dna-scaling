import json,pathlib,unittest
import torch
import train_cnn_evaluator as evaluator

class CnnEvaluatorTests(unittest.TestCase):
 def test_published_architecture_and_encoding(self):
  torch.set_num_threads(2)
  model=evaluator.upstream.DilatedResNet1D().eval()
  x=torch.stack([evaluator.onehot('ACGT'*128),evaluator.onehot('TGCA'*128)])
  with torch.no_grad():output=model(x)
  self.assertEqual(tuple(x.shape),(2,4,512));self.assertEqual(tuple(output.shape),(2,));self.assertTrue(torch.isfinite(output).all())
  self.assertTrue(torch.equal(x[0].flip((0,1)),evaluator.onehot(('ACGT'*128).translate(str.maketrans('ACGT','TGCA'))[::-1])))
 def test_source_groups_do_not_cross_folds(self):
  rows=[json.loads(line) for line in (evaluator.DATA/'prompts.jsonl').read_text().splitlines()]
  groups={f:{r['group_id'] for r in rows if r['fold']==f} for f in ('train','calibration','test')}
  self.assertEqual([len(groups[f]) for f in groups],[60,20,20])
  self.assertFalse(groups['train']&groups['test'] or groups['train']&groups['calibration'] or groups['calibration']&groups['test'])
  for row in rows:
   self.assertEqual(len(row['natural']),512);self.assertEqual(len(row['prefix']),256);self.assertEqual(len(row['null_natural']),512)
 def test_continuations_only_and_matched_labels(self):
  prompts=[{'id':0,'group_id':'g','fold':'train','category':'c','prefix':'A'*256,'natural':'C'*512,'null_natural':'G'*512}]
  generated={size:{0:{'generated':base*512}} for size,base in [('5m','A'),('10m','T')]}
  rows=evaluator.dataset(prompts,generated,'main');self.assertEqual(len(rows),3)
  self.assertEqual([r['label'] for r in rows],[0,1,1]);self.assertTrue(all(len(r['text'])==512 for r in rows))
if __name__=='__main__':unittest.main()
