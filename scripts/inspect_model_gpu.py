import json
import pathlib
import torch
from savanna.arguments import GlobalConfig
from savanna.initialize import initialize_megatron
from savanna.training import get_model

if __name__ == "__main__":
    config = GlobalConfig.consume_global_config()
    config.configure_distributed_args()
    config.build_tokenizer()
    initialize_megatron(global_config=config)
    if torch.distributed.get_world_size() != 1:
        raise ValueError("Parameter inspection requires exactly one GPU/rank")
    model = get_model(global_config=config, use_cache=False)
    report = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "padded_vocab_size": config.padded_vocab_size,
        "parameters_by_top_level_module": {
            name: sum(p.numel() for p in module.parameters())
            for name, module in model.named_children()
        },
    }
    path = pathlib.Path(config.save).parent / (
        pathlib.Path(config.save).name + "-parameters.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()
