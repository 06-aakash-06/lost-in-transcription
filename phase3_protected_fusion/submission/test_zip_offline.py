"""Run exact extracted ZIP in an empty HF cache with networking blocked."""
import argparse,hashlib,json,os,subprocess,sys,zipfile,shutil,time
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent

def test(archive,label,repeat=1):
    expected=archive.with_suffix('.zip.sha256').read_text().split()[0]
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    work=HERE/'results/offline_zip_tests'/label;work.mkdir(parents=True,exist_ok=True)
    source=work/'extracted';source.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;z.extractall(source)
    manifest=pd.read_csv(ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv',sep='\t',keep_default_na=False)
    ids=['71ac896ea0bd4999ac7dad89c7298da1','3fc1b82ff2e64625af2a92801b63cc8f']
    samples=manifest.set_index('clip_id').loc[ids].reset_index()
    assert samples.duration_s.max()>30
    data=work/'data';(data/'clips').mkdir(parents=True,exist_ok=True)
    names=['short,"quoted".mp3','long.mp3']
    for name,row in zip(names,samples.itertuples()):shutil.copyfile(ROOT/row.path,data/'clips'/name)
    # Reverse requested order so accidental sorting is detected.
    pd.DataFrame({'audio_filename':names[::-1]}).to_csv(data/'test_metadata.csv',index=False)
    env=os.environ.copy();env.update(HF_HOME=str(work/'empty_hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LIT_DATA_DIR=str(data),LIT_SUBMISSION_PATH=str(work/'submission.csv'),PYTHONPATH='')
    env.update(HF_HUB_CACHE=str(work/'empty_hf_cache/hub'),HUGGINGFACE_HUB_CACHE=str(work/'empty_hf_cache/hub'),HF_MODULES_CACHE=str(work/'empty_hf_cache/modules'))
    script="""import socket,runpy

def blocked(*args,**kwargs):raise RuntimeError('NETWORK BLOCKED')
socket.socket.connect=blocked
socket.socket.connect_ex=blocked
socket.create_connection=blocked
runpy.run_path('main.py',run_name='__main__')
"""
    start=time.monotonic()
    first=None
    for attempt in range(repeat):
        with (work/f'inference_{attempt+1}.log').open('w') as log:subprocess.run([sys.executable,'-c',script],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw=(work/'submission.csv').read_bytes()
        if first is None:first=raw
        else:assert raw==first,'inference output is nondeterministic'
    elapsed=time.monotonic()-start
    meta=pd.read_csv(data/'test_metadata.csv',keep_default_na=False);output=pd.read_csv(work/'submission.csv',keep_default_na=False)
    assert list(output.columns)==['audio_filename','transcript'] and len(output)==len(meta)
    assert output.audio_filename.tolist()==meta.audio_filename.tolist()
    assert output.transcript.map(lambda x:isinstance(x,str) and bool(x.strip())).all() and not output.isna().any().any()
    long_reference=samples.iloc[1].text
    long_prediction=output[output.audio_filename=='long.mp3'].iloc[0].transcript
    assert long_prediction.split()[-8:]==long_reference.split()[-8:],'native long tail missing'
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    result={'zip':str(archive),'sha256':expected,'clips':2,'native_long_form_seconds':float(samples.duration_s.max()),'row_order':True,'csv_quoting':True,'nonempty_strings':True,'offline':True,'empty_hf_cache':True,'socket_connections_blocked':True,'elapsed_s':elapsed,'python':sys.version,'device':'mps','official_cuda_container_tested':False,'long_clip_reference_tail_8_words_retained':True}
    result.update(repeat_runs=repeat,deterministic_output=repeat>1)
    (work/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--repeat',type=int,default=1);a=p.parse_args();test(a.zip.resolve(),a.label,a.repeat)
