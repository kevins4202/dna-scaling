import pathlib,json,sys
root=pathlib.Path(sys.argv[1])
for p in (root/'configs').rglob('*.yml'):
 s=p.read_text().replace('1e-06','0.000001').replace('1e-08','0.00000001').replace('6e-05','0.00006')
 p.write_text(s)
p=root/'configs/data.yml'; c=json.loads(p.read_text());c['train-data-weights']=[1.0];p.write_text(json.dumps(c,indent=2)+'\n')
p=root/'scripts/build_models.py'
s=p.read_text().replace('json.dumps(c, indent=2)',"json.dumps(c, indent=2).replace('1e-06', '0.000001').replace('1e-08', '0.00000001')").replace('json.dumps(smoke, indent=2)',"json.dumps(smoke, indent=2).replace('1e-06', '0.000001').replace('1e-08', '0.00000001')")
p.write_text(s)
