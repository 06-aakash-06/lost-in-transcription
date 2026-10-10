"""One deterministic recommended archive and optional standalone backup."""
import argparse,json,hashlib,os,shutil,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SKIP={'merge_validation.json','adapter_model.safetensors','adapter_config.json','training_state.pt','optimizer.pt','scheduler.pt','trainer_state.json'}

def build(config_path,output):
    if output.exists():raise FileExistsError('preserving existing archive')
    config=json.loads(config_path.read_text());entries={'main.py':HERE/'main.py','config.json':config_path,'MODEL_NOTICES.md':HERE/'MODEL_NOTICES.md','WHISPER_LICENSE.txt':ROOT/'phase1_whisper_anchor/WHISPER_LICENSE.txt','requirements-inference.txt':HERE/'requirements-inference.txt'}
    apache=HERE/'APACHE_LICENSE.txt'
    if apache.exists():entries['APACHE_LICENSE.txt']=apache
    for path in sorted((HERE/'runtime').glob('*.py')):entries['runtime/'+path.name]=path
    sources={}
    model_bytes=0
    for name,model in config['models'].items():
        path=Path(model.pop('source'))
        if not (path/'config.json').exists() or not list(path.glob('*.safetensors')):raise FileNotFoundError('missing merged model')
        for asset in sorted(path.iterdir()):
            if asset.is_file() and asset.name not in SKIP:
                entries[f'models/{name}/{asset.name}']=asset;model_bytes+=asset.stat().st_size
        sources[name]=str(path.resolve())
    entries['config.json']=(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.zip.tmp');inventory={}
    try:
        with zipfile.ZipFile(temporary,'w',allowZip64=True) as archive:
            for name,item in sorted(entries.items()):
                info=zipfile.ZipInfo(name,(2020,1,1,0,0,0));info.external_attr=0o644<<16
                info.compress_type=zipfile.ZIP_STORED if isinstance(item,Path) and item.stat().st_size>16_000_000 else zipfile.ZIP_DEFLATED
                digest=hashlib.sha256();size=0
                with archive.open(info,'w',force_zip64=True) as dest:
                    if isinstance(item,Path):
                        with item.open('rb') as src:
                            while block:=src.read(8*1024*1024):dest.write(block);digest.update(block);size+=len(block)
                    else:dest.write(item);digest.update(item);size=len(item)
                inventory[name]={'sha256':digest.hexdigest(),'bytes':size,'source':str(item.resolve()) if isinstance(item,Path) else None}
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
            assert 'main.py' in archive.namelist()
        with temporary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        os.replace(temporary,output)
        output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
        output.with_suffix('.inventory.json').write_text(json.dumps({'sha256':digest,'zip_bytes':output.stat().st_size,'model_bytes':model_bytes,'model_sources':sources,'entries':inventory},indent=2)+'\n')
        print(json.dumps({'zip':str(output.resolve()),'sha256':digest,'bytes':output.stat().st_size,'model_bytes':model_bytes}),flush=True)
    finally:temporary.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.config,a.output)
