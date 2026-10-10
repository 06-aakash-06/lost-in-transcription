"""Verify prior overnight outputs and existing competition archives unchanged."""
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def check():
    started=time.time();base=ROOT/'overnight_attack';manifest=json.loads((base/'PRESERVATION_MANIFEST.json').read_text())
    changes=[]
    for record in manifest['files']:
        path=base/record['path']
        if not path.is_file() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:changes.append(record['path'])
    archives={}
    for path in sorted((ROOT/'artifacts').glob('*.zip.sha256')):
        archive=path.with_suffix('');expected=path.read_text().split()[0]
        if archive.name.startswith('FINAL_MAX_WER_REDUCTION') or archive.name.startswith('FINAL_LARGEV3_STANDALONE'):continue
        actual=digest(archive);archives[archive.name]={'expected':expected,'actual':actual,'unchanged':actual==expected}
    r4=ROOT/'phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218'
    stopped={'adapter_model.safetensors':'987a885a7f53e956e3d02ebace21411d88f029be8a28cfcc1116c06729abe7cf','training_state.pt':'f14acf6db9444e3625a387b5cf1feb96601fd1dfd292a3210010422f84e597ad'}
    r4_hashes={name:digest(r4/name) for name in stopped}
    output={'overnight_files_checked':len(manifest['files']),'changed_overnight_files':changes,'existing_archives':archives,'stopped_ratio4':{'checkpoint':str(r4),'batch':218,'hashes':r4_hashes,'unchanged':r4_hashes==stopped},'elapsed_seconds':time.time()-started}
    output['pass']=not changes and all(v['unchanged'] for v in archives.values()) and r4_hashes==stopped
    (OUT/'preservation_verified.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
    if not output['pass']:raise RuntimeError('Prior artifact verification requires investigation')
if __name__=='__main__':check()
