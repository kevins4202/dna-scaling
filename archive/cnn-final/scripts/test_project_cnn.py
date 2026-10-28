import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch

import prepare_project_cnn as prepare
import score_project_cnn as scoring
import train_project_cnn as training
import evaluate_project_cnn as evaluation


class Dummy(torch.nn.Module):
    def forward(self, x):
        return x[:, 0].mean(1) + 2*x[:, 1].mean(1)


class ProjectCnnTests(unittest.TestCase):
    def test_chunk_scoring_and_reverse_complement(self):
        model = Dummy()
        text = 'A'*512+'C'*512
        mean, chunks = scoring.score_text(model, text, 'cpu')
        self.assertEqual(chunks, [.5, 1.])
        self.assertEqual(mean, .75)
        rc = text.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        self.assertEqual(scoring.score_text(model, rc, 'cpu')[0], mean)

    def test_invalid_dna_rejected(self):
        for text in ('A'*511, 'A'*513, 'N'*512, 'a'*512, '', None):
            with self.assertRaises(ValueError):
                scoring.score_text(Dummy(), text, 'cpu')

    def test_modified_release_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root/'model.pt').write_bytes(b'changed')
            (root/'manifest.json').write_text(json.dumps({'version': 'cnn-v1', 'frozen': True,
                'files_sha256': {'model.pt': hashlib.sha256(b'original').hexdigest()}}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                scoring.load_release(root, 'cpu')

    def test_pair_aggregation_and_metric_direction(self):
        rows = [{'tag': 'a', 'pair_id': str(i//2)} for i in range(4)]
        metrics, _ = training.paired_metrics(rows, [0, 0, 1, 1], [2, 2, 3, 3])
        self.assertEqual(metrics['macro_auc'], 1.)
        self.assertEqual(metrics['pairs'], 2)

    def test_mixture_weighting_does_not_follow_sample_count(self):
        rows = [{'group_id':str(i),'category':'metagenomes','natural_logit':0.,'generated_logit':1.} for i in range(20)]
        rows += [{'group_id':'euk','category':'eukaryotic_genic_windows','natural_logit':1.,'generated_logit':0.}]
        result = evaluation.metrics(rows)
        self.assertAlmostEqual(result['mixture_weighted_stratified_auc'],24/59)
        self.assertEqual(result['represented_reference_weight'],59)

    def test_prepared_split_and_content(self):
        manifest = json.loads((prepare.OUT/'manifest.json').read_text())
        seen_tags, seen_sequences = set(), set()
        for fold in ('train', 'calibration', 'test'):
            path = prepare.OUT/f'{fold}.jsonl'
            self.assertEqual(training.digest(path), manifest['file_sha256'][path.name])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            tags = {r['tag'] for r in rows}
            self.assertFalse(tags & seen_tags)
            keys = set()
            for row in rows:
                self.assertEqual(row['fold'], fold)
                self.assertGreaterEqual(row['offset'], row['excluded_prefix'])
                self.assertGreaterEqual(row['excluded_prefix'], 4096)
                for field in ('natural', 'generated', 'null_natural'):
                    self.assertEqual(len(row[field]), 512)
                    self.assertLessEqual(set(row[field]), set('ACGT'))
                    keys.add(prepare.canonical(row[field]))
            self.assertFalse(keys & seen_sequences)
            seen_tags.update(tags)
            seen_sequences.update(keys)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
