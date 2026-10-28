"""Record the score-ordering correction without changing registered evidence."""
from common import ROOT,OUT,ORIGINAL,guard,read,rows,sha,write
p=guard();prompts=rows(ORIGINAL/'prompts.jsonl');scores=read(OUT/'released-7b-scores.json')['predictions']
assert len(prompts)==len(scores)==100
expected={str(r['id']):(r['group_id'],r['category']) for r in prompts}
observed={str(r['id']):(r['group_id'],r['category']) for r in scores}
assert len(expected)==len(observed)==100 and expected==observed
write(OUT/'validation-correction.json',{'reason':'Original validator zipped numeric prompt order against lexicographically sorted score ids. All unique ids, categories and groups match exactly when joined by id.','registered_protocol_sha256':sha(OUT/'protocol.json'),'original_validator_sha256':sha(ROOT/'scripts/baseline/validate.py'),'corrected_validator_sha256':sha(ROOT/'scripts/baseline/validate_final.py'),'original_prompt_ids_first12':[r['id'] for r in prompts[:12]],'score_ids_first12':[r['id'] for r in scores[:12]],'identities_match_by_unique_id':True,'scientific_inputs_and_results_unchanged':True})
print('VALIDATION_ORDERING_CORRECTION_RECORDED')
