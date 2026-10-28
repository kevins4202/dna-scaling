"""CPU checks for the strict reload guard and official cache-offset convention."""
import types,unittest
import torch
import generate as runner

class BaselineGuards(unittest.TestCase):
    def test_cache_offsets_follow_prefill(self):
        cache={name:types.SimpleNamespace(seqlen_offset=0) for name in ('mha','hcl','hcm','hcs')}
        runner.advance(cache,256)
        self.assertEqual([c.seqlen_offset for c in cache.values()],[256]*4)
        runner.advance(cache,256)
        self.assertEqual([c.seqlen_offset for c in cache.values()],[257]*4)

    def test_strict_reload_rejects_missing_unexpected_and_shape_changes(self):
        model=types.SimpleNamespace(state_dict=lambda:{'weight':torch.zeros(2,3)})
        previous=runner.checked_load.original
        runner.checked_load.original=lambda self,state_dict,strict:strict
        try:
            self.assertTrue(runner.checked_load(model,{'weight':torch.ones(2,3)}))
            self.assertTrue(runner.checked_load(model,{'weight':torch.ones(2,3),'projection._extra_state':None}))
            for state in ({},{'weight':torch.ones(3,2)},{'weight':torch.ones(2,3),'unknown.weight':torch.ones(1)}):
                with self.assertRaises(AssertionError):runner.checked_load(model,state)
        finally:runner.checked_load.original=previous

    def test_duplicate_output_must_match_tied_input_embedding(self):
        model=types.SimpleNamespace(state_dict=lambda:{'embedding_layer.weight':torch.zeros(2,3)},config=types.SimpleNamespace(tie_embeddings=True))
        previous=runner.checked_load.original
        runner.checked_load.original=lambda self,state_dict,strict:strict
        try:
            good={'embedding_layer.weight':torch.ones(2,3),'unembed.weight':torch.ones(2,3)}
            self.assertTrue(runner.checked_load(model,good))
            self.assertTrue(runner.checked_load.evidence['duplicate_tied_output_verified'])
            with self.assertRaises(AssertionError):runner.checked_load(model,{**good,'unembed.weight':torch.zeros(2,3)})
            model.config.tie_embeddings=False
            with self.assertRaises(AssertionError):runner.checked_load(model,good)
        finally:runner.checked_load.original=previous

    def test_attention_adapter_preserves_kernel_outputs(self):
        from compat import FlashAttention26Adapter
        original=tuple(object() for _ in range(8))
        result=FlashAttention26Adapter.forward_outputs(original)
        self.assertEqual(result,tuple(original[i] for i in (0,5,6,7)))
        self.assertEqual(FlashAttention26Adapter.forward_outputs(result),result)
        with self.assertRaises(AssertionError):FlashAttention26Adapter.forward_outputs(original[:6])

if __name__=='__main__':unittest.main()

