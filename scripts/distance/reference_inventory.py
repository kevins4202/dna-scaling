"""Inspect pinned upstream evaluation shards and metadata before selecting new DNA."""
import gzip,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import sample_opengenome as source
OUT=ROOT/'reports/category-uncertainty-v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True);result={}
    for category in source.WEIGHTS:
        if category=='promoters':continue
        files=source.list_files(category,'test');shard=files[0]
        with source.request(f'https://huggingface.co/datasets/{source.REPO}/resolve/{source.REVISION}/'+shard['path']) as response,gzip.GzipFile(fileobj=response) as stream:
            rows=[]
            for number,line in enumerate(stream,1):
                row=json.loads(line);rows.append({'row':number,'length':len(row['text']),'metadata':{k:v for k,v in row.items() if k!='text'}})
                if number==3:break
        result[category]={'test_shards':len(files),'files':files,'first_records':rows};print(category,len(files),rows,flush=True)
    (OUT/'upstream-test-inventory.json').write_text(json.dumps({'revision':source.REVISION,'categories':result},indent=2)+'\n')
if __name__=='__main__':main()
