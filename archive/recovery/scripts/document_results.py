import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
p=root/'scripts/audit_dataset.py';s=p.read_text().replace("'reference_characters_per_category':2000000", "'reference_characters_per_category':{k:v['independent_reference']['chars'] for k,v in comparison.items()}")
s=s.replace('Reference uses disjoint shards where possible, otherwise disjoint rows after the original scan.', 'Reference uses disjoint shards where possible, otherwise disjoint source coordinates; GTDB was broadened to 16 disjoint shards and eukaryotic reference to all 40 shards.')
p.write_text(s)
p=root/'reports/distribution-audit.json'
if p.exists():
 c=json.loads(p.read_text());c['reference_characters_per_category']={k:v['independent_reference']['chars'] for k,v in c['categories'].items()};c['note']='Independent reference has 8M characters from 16 disjoint GTDB shards and 2M per other category. Eukaryotic reference spans all 40 shards in regions disjoint from training. These are bounded samples, not full-population statistics. Thresholds are screening rules, not equivalence tests.';p.write_text(json.dumps(c,indent=2)+'\n')
