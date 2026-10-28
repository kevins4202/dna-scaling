import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
p=root/'configs/data.yml'
c=json.loads(p.read_text());c.pop('use_wandb',None)
p.write_text(json.dumps(c,indent=2)+'\n')
