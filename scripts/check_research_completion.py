import hashlib, json, math, pathlib, re, sys

root = pathlib.Path(__file__).resolve().parents[1]
size = sys.argv[1]
log = (root / f"logs/research/{size}-train.log").read_text()
lines = [x for x in log.splitlines() if re.search(r"iteration\s+\d+/\s+3051", x)]
assert lines, "No training metrics logged"
for line in log.splitlines():
    match = re.search(r"lm_loss:\s*([\w.+-]+)", line)
    if match:
        assert math.isfinite(float(match.group(1)))
for value in re.findall(r"number of nan iterations:\s*(\d+)", log):
    assert int(value) == 0
verification = json.loads(
    (root / f"reports/research/{size}-verification.json").read_text()
)
assert verification["checkpoint_training_steps"] == 3051
assert (
    verification["strict_state_reload"]
    and verification["finite_logits"]
    and verification["new_tokens"] == 32
)
checkpoint = root / f"checkpoints/{size}/global_step3051/mp_rank_00_model_states.pt"
assert checkpoint.stat().st_size > 0
report = {
    "model": size,
    "training_steps": 3051,
    "processed_training_tokens": 99975168,
    "final_training_log": lines[-1],
    "final_checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "checkpoint_reload_and_generation": verification,
    "status": "verified_complete",
}
(root / f"reports/research/{size}-complete.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(size, "VERIFIED_COMPLETE")
