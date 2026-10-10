"""Build one named final archive from the evaluated selection, offline only."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'phase4_stronger_anchor'
OUT = HERE / 'results'
sys.path.insert(0,str(ROOT))
SKIP = {'merge_validation.json','adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt',
        'training_state.pt','trainer_state.json','training_args.bin','train_metrics.json','average_provenance.json'}

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def build(output):
    if output.exists():raise FileExistsError(f'refusing to overwrite archive: {output}')
    selection=json.loads((OUT/'fusion_selection.json').read_text())
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.zip.tmp')
    if not selection['deploy_new_fusion']:
        previous=ROOT/'artifacts/FINAL_SUBMISSION_CONSERVATIVE.zip'
        expected=previous.with_suffix('.zip.sha256').read_text().split()[0]
        assert digest(previous)==expected
        shutil.copyfile(previous,temporary);assert digest(temporary)==expected
        os.replace(temporary,output)
        output.with_suffix('.zip.sha256').write_text(f'{expected}  {output.name}\n')
        result={'selection':'previous_conservative_retained','zip':str(output.resolve()),'sha256':expected,
            'reason':'New fusion did not pass the OOF improvement and conversation-protection gate.'}
        (OUT/'final_build.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return result
    model=HERE/'models/selected_anchor'
    if not (model/'merge_validation.json').exists():raise FileNotFoundError('selected new anchor has not passed merge validation')
    assert json.loads((model/'merge_validation.json').read_text())['equivalent']
    voter=selection['selected_fusion']['voter']
    if voter=='R3_half':support=ROOT/'phase1_whisper_anchor/models/final_merged'
    elif voter=='R3_full':support=ROOT/'phase3_protected_fusion/models/whisper_full'
    else:support=model
    config={'anchor_model':'whisper_anchor','anchor_language':selection['anchor_language'],
        'third_model':'whisper_anchor' if support==model else 'whisper_support',
        'third_language':voter if voter in ['indonesian','javanese','auto'] else 'indonesian',
        'cuda_batch_size':8,'fusion':selection['selected_fusion']['config']}
    main=(ROOT/'phase3_protected_fusion/main.py').read_text()
    main=main.replace("language='indonesian',batch_size=batch_size,long_mode='native')",
        "language=(None if config.get('anchor_language')=='auto' else config.get('anchor_language','indonesian')),batch_size=batch_size,long_mode='native')",1)
    main=main.replace("language='indonesian',batch_size=batch_size,long_mode='native')",
        "language=(None if config.get('third_language')=='auto' else config.get('third_language','indonesian')),batch_size=batch_size,long_mode='native')",1)
    compile(main,'main.py','exec')
    entries={'main.py':main.encode(),'config.json':(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()}
    for path in sorted((ROOT/'phase3_protected_fusion/runtime').glob('*.py')):
        if path.name!='edit_gate.py':entries['runtime/'+path.name]=path
    models={'whisper_anchor':model,'meralion':ROOT/'phase2_meralion/models/final_merged'}
    if support!=model:models['whisper_support']=support
    for name,folder in models.items():
        if not (folder/'config.json').exists() or not list(folder.glob('*.safetensors')):raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name not in SKIP:entries[f'models/{name}/{path.name}']=path
    entries['WHISPER_LICENSE.txt']=ROOT/'phase1_whisper_anchor/WHISPER_LICENSE.txt'
    entries['MODEL_NOTICES.md']=ROOT/'phase3_protected_fusion/MODEL_NOTICES.md'
    try:
        with zipfile.ZipFile(temporary,'w') as archive:
            for name,item in sorted(entries.items()):
                info=zipfile.ZipInfo(name,(2020,1,1,0,0,0));info.external_attr=0o644<<16
                info.compress_type=zipfile.ZIP_STORED if isinstance(item,Path) and item.stat().st_size>16_000_000 else zipfile.ZIP_DEFLATED
                if isinstance(item,Path):
                    with archive.open(info,'w',force_zip64=True) as dest,item.open('rb') as src:shutil.copyfileobj(src,dest,1024*1024)
                else:archive.writestr(info,item)
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
            assert 'main.py' in archive.namelist()
            assert not any(Path(n).name in SKIP for n in archive.namelist())
            assert not any(n.endswith(('.csv','.tsv','.mp3','.wav','.pt')) for n in archive.namelist())
        checksum=digest(temporary);os.replace(temporary,output)
        output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n')
        result={'selection':'new_ratio4_anchor','checkpoint':selection['checkpoint'],'language':selection['anchor_language'],
            'voter':voter,'config':config,'zip':str(output.resolve()),'sha256':checksum,'size_bytes':output.stat().st_size,
            'oof_metrics':selection['selected_fusion']['metrics']}
        (OUT/'final_build.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True);return result
    finally:temporary.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/FINAL_SUBMISSION_STRONGER_ANCHOR.zip');a=p.parse_args()
    build(a.output)
