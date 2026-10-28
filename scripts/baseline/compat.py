"""Process-local Vortex adapter for the installed FlashAttention 2.6 ABI."""
class FlashAttention26Adapter:
    def __init__(self,backend):self.backend=backend
    def __getattr__(self,name):return getattr(self.backend,name)
    @staticmethod
    def forward_outputs(result):
        if len(result)==4:return result
        assert len(result)==8,"Unknown FlashAttention forward ABI"
        return result[0],result[5],result[6],result[7]
    def fwd(self,*args,**kwargs):return self.forward_outputs(self.backend.fwd(*args,**kwargs))
    def varlen_fwd(self,*args,**kwargs):return self.forward_outputs(self.backend.varlen_fwd(*args,**kwargs))
def install():
    import flash_attn
    import vortex.ops.attn_interface as interface
    assert flash_attn.__version__=="2.6.3",flash_attn.__version__
    interface.flash_attn_gpu=FlashAttention26Adapter(interface.flash_attn_gpu)
    return {"flash_attention":flash_attn.__version__,"adapter":"forward tuple (out,q,k,v,out_padded,lse,mask,rng) to (out,lse,mask,rng); cached kernel unchanged"}
