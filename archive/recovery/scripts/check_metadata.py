import csv,gzip,json,pathlib,urllib.request
root=pathlib.Path('/home/ksong37/cis6200/data')
url='https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/84d2a7e690c8bb395d1f4868822dfa031ea561c6/species_metadata.csv'
p=root/'species_metadata.csv'
if not p.exists():
 with urllib.request.urlopen(url,timeout=90) as response,p.open('wb') as out:
  while block:=response.read(1024*1024):out.write(block)
with p.open() as f:
 reader=csv.DictReader(f)
 print('METADATA_COLUMNS',reader.fieldnames)
 for _,row in zip(range(3),reader):print('METADATA_EXAMPLE',json.dumps(row))
for p in list((root/'source-shards').glob('*'))[:3]:
 with gzip.open(p,'rt') as f:row=json.loads(f.readline())
 print('SHARD',p.name,'FIELDS',list(row),'NON_TEXT_FIELDS',{k:v for k,v in row.items() if k!='text'})
