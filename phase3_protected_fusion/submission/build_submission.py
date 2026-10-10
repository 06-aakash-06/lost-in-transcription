"""Deterministic ZIP64 archive, merged weights only; never changes Phase 1 ZIP."""
import os,json,hashlib,shutil,zipfile,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
SKIP={'merge_validation.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','training_args.bin','trainer_state.json'}
def build(output,config):
    if output.resolve()==(ROOT/'artifacts/phase1_whisper_anchor_submission.zip').resolve():raise ValueError('refusing to overwrite known-good Phase 1 archive')
    entries={'main.py':HERE/'main.py','config.json':(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()}
    for path in sorted((HERE/'runtime').glob('*.py')):
        if path.name!='edit_gate.py':entries['runtime/'+path.name]=path
    models=[('whisper',ROOT/'phase1_whisper_anchor/models/final_merged'),('meralion',ROOT/'phase2_meralion/models/final_merged')]
    extra={config.get('anchor_model','whisper'),config.get('third_model')}-{'whisper','meralion',None}
    for name in sorted(extra):models.append((name,HERE/'models'/name))
    for name,model in models:
        if not (model/'config.json').exists() or not list(model.glob('*.safetensors')):raise FileNotFoundError('merged checkpoint missing')
        for path in sorted(model.iterdir()):
            if path.is_file() and path.name not in SKIP:entries[f'models/{name}/{path.name}']=path
    entries['WHISPER_LICENSE.txt']=ROOT/'phase1_whisper_anchor/WHISPER_LICENSE.txt'
    entries['MODEL_NOTICES.md']=HERE/'MODEL_NOTICES.md'
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary,'w') as archive:
            for name,item in sorted(entries.items()):
                info=zipfile.ZipInfo(name,(2020,1,1,0,0,0));info.external_attr=0o644<<16
                info.compress_type=zipfile.ZIP_STORED if isinstance(item,Path) and item.stat().st_size>16_000_000 else zipfile.ZIP_DEFLATED
                if isinstance(item,Path):
                    with archive.open(info,'w',force_zip64=True) as dest,item.open('rb') as src:shutil.copyfileobj(src,dest,1024*1024)
                else:archive.writestr(info,item)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
            assert 'main.py' in archive.namelist()
            assert not any(Path(n).name in SKIP for n in archive.namelist())
        digest=hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()
        os.replace(temporary,output);output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
        return digest
    finally:temporary.unlink(missing_ok=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/phase3_protected_fusion_submission.zip');p.add_argument('--config',type=Path,default=HERE/'configs/candidate_A.json');a=p.parse_args()
    print(build(a.output,json.loads(a.config.read_text())))
