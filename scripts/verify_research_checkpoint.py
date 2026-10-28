import json
import pathlib
import torch
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.training import get_model


if __name__ == "__main__":
    c = GlobalConfig.consume_global_config()
    c.configure_distributed_args()
    c.build_tokenizer()
    initialize_megatron(global_config=c)
    model = get_model(global_config=c, use_cache=False)
    checkpoint = (
        pathlib.Path(c.save) / f"global_step{c.train_iters}/mp_rank_00_model_states.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert (
        state["iteration"] == c.train_iters
    ), "Checkpoint is not from final training step"
    model.load_state_dict(state["module"], strict=True)
    model = model.cuda().to(torch.bfloat16).eval()
    prefix = "ACGTACGTTGCAACGT"
    tokens = torch.tensor(
        [c.tokenizer.tokenize(prefix)], device="cuda", dtype=torch.long
    )
    allowed = torch.tensor([ord(x) for x in "ACGT"], device="cuda")
    generated = ""
    masses = []
    torch.manual_seed(6200)
    with torch.inference_mode():
        for _ in range(32):
            positions = torch.arange(c.seq_length, device="cuda")[None, :]
            padded = torch.nn.functional.pad(
                tokens, (0, c.seq_length - tokens.shape[1]), value=0
            )
            output = model((padded, None, positions))
            if isinstance(output, tuple):
                output = output[0]
            logits = output[:, tokens.shape[1] - 1, :].float()
            assert torch.isfinite(logits).all(), "Nonfinite logits"
            masses.append(torch.softmax(logits, dim=-1)[0, allowed].sum().item())
            nxt = allowed[
                torch.multinomial(torch.softmax(logits[0, allowed], dim=-1), 1)
            ].reshape(1, 1)
            generated += chr(nxt.item())
            tokens = torch.cat([tokens, nxt], dim=1)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_training_steps": state["iteration"],
        "strict_state_reload": True,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "prefix": prefix,
        "generated": generated,
        "new_tokens": len(generated),
        "finite_logits": True,
        "sampling": "Seeded sampling restricted to A/C/G/T; engineering verification only, not a biological quality evaluation",
        "mean_unconstrained_probability_mass_on_acgt": sum(masses) / len(masses),
    }
    out = pathlib.Path("../../reports/research") / (
        pathlib.Path(c.save).name + "-verification.json"
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()
