"""Test exact ZIP contents offline, without duplicating large model storage."""
import os,sys,json,hashlib,zipfile,argparse,subprocess,time,shutil
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def validate(archive,label,repeat=2):
    inventory=json.loads(archive.with_suffix('.inventory.json').read_text());assert digest(archive)==inventory['sha256']
    work=HERE/'results/offline_zip_tests'/label;source=work/'extracted';source.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;assert set(z.namelist())==set(inventory['entries'])
        for name,record in inventory['entries'].items():
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            # Every archive byte is independently hashed. Model hardlinks are
            # an extraction-storage optimization, not an external dependency.
            with z.open(name) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
            if name.startswith('models/') and record['source']:
                original=Path(record['source']);assert digest(original)==record['sha256']
                if not target.exists():os.link(original,target)
            else:target.write_bytes(z.read(name))
            assert digest(target)==record['sha256']
    config=json.loads((source/'config.json').read_text())
    for name in config['models']:
        assets=source/'models'/name;assert (assets/'config.json').exists()
        indexes=list(assets.glob('*.safetensors.index.json'))
        for index in indexes:
            mapping=json.loads(index.read_text())['weight_map'];assert all((assets/n).is_file() for n in set(mapping.values()))
        assert 'source' not in config['models'][name]
    manifest=pd.read_csv(ROOT/'phase1_whisper_anchor/data/fold_A_valid.tsv',sep='\t',keep_default_na=False)
    clips=manifest.set_index('clip_id').loc[['71ac896ea0bd4999ac7dad89c7298da1','3fc1b82ff2e64625af2a92801b63cc8f']].reset_index()
    assert clips.iloc[1].duration_s>30
    data=work/'data';(data/'clips').mkdir(parents=True,exist_ok=True)
    names=['short,"quoted".mp3','long.mp3']
    for name,row in zip(names,clips.itertuples()):shutil.copyfile(ROOT/row.path,data/'clips'/name)
    pd.DataFrame({'audio_filename':names[::-1]}).to_csv(data/'test_metadata.csv',index=False)
    env=os.environ.copy();env.update(HF_HOME=str(work/'empty_hf_cache'),HF_HUB_CACHE=str(work/'empty_hf_cache/hub'),HUGGINGFACE_HUB_CACHE=str(work/'empty_hf_cache/hub'),HF_MODULES_CACHE=str(work/'empty_hf_cache/modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LIT_DATA_DIR=str(data),LIT_SUBMISSION_PATH=str(work/'submission.csv'),PYTHONPATH='')
    script="""import socket,runpy,importlib.util,builtins,os,time,json
original_spec=importlib.util.find_spec
def filtered_spec(name,package=None):
    if name.split('.')[0] in ('peft','sklearn'):return None
    return original_spec(name,package)
importlib.util.find_spec=filtered_spec
original_import=builtins.__import__
def filtered_import(name,*args,**kwargs):
    if name.split('.')[0] in ('peft','sklearn'):raise ImportError('OPTIONAL TRAINING DEPENDENCY BLOCKED')
    return original_import(name,*args,**kwargs)
builtins.__import__=filtered_import
def blocked(*args,**kwargs):raise RuntimeError('NETWORK BLOCKED')
socket.socket.connect=blocked
socket.socket.connect_ex=blocked
socket.create_connection=blocked
from pathlib import Path
from runtime.whisper import Whisper
from runtime.meralion import Meralion
timings={}
def instrument(cls):
    init=cls.__init__;transcribe=cls.transcribe_arrays
    def traced_init(self,*args,**kwargs):
        started=time.monotonic();init(self,*args,**kwargs)
        self._local_validation_name=Path(args[0]).name
        timings[self._local_validation_name]={'load_seconds':time.monotonic()-started,'decode_seconds':0.,'clips':0}
    def traced_decode(self,waves):
        started=time.monotonic();result=transcribe(self,waves)
        item=timings[self._local_validation_name];item['decode_seconds']+=time.monotonic()-started;item['clips']+=len(waves)
        return result
    cls.__init__=traced_init;cls.transcribe_arrays=traced_decode
instrument(Whisper);instrument(Meralion)
try:runpy.run_path('main.py',run_name='__main__')
finally:Path(os.environ['LIT_LOCAL_VALIDATION_TIMINGS']).write_text(json.dumps(timings))
"""
    times=[];first=None
    for attempt in range(repeat):
        env['LIT_LOCAL_VALIDATION_TIMINGS']=str(work/f'component_times_{attempt+1}.json')
        started=time.monotonic()
        with (work/f'inference_{attempt+1}.log').open('w') as log:subprocess.run([sys.executable,'-c',script],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        times.append(time.monotonic()-started);raw=(work/'submission.csv').read_bytes()
        if first is None:first=raw
        else:assert raw==first,'repeated inference differs'
        assert not (work/f'inference_{attempt+1}.log').read_text().strip(),'unexpected inference logging'
    output=pd.read_csv(work/'submission.csv',keep_default_na=False);metadata=pd.read_csv(data/'test_metadata.csv',keep_default_na=False)
    assert list(output.columns)==['audio_filename','transcript'];assert len(output)==len(metadata)
    assert output.audio_filename.tolist()==metadata.audio_filename.tolist();assert not output.isna().any().any()
    assert output.transcript.map(lambda s:isinstance(s,str) and bool(s.strip())).all()
    # Verify genuine native long-form coverage against tail words known to be
    # after 30s, permitting recognition errors instead of requiring perfect ASR.
    tail=clips.iloc[1].text.split()[-8:];long_text=output.loc[output.audio_filename=='long.mp3','transcript'].iloc[0].split()
    overlap=sum(w in long_text[-16:] for w in tail)
    assert overlap>=5,'native long-form tail not retained'
    # Reordering cannot alter any other clip's result.
    pd.DataFrame({'audio_filename':names}).to_csv(data/'test_metadata.csv',index=False)
    started=time.monotonic()
    env['LIT_LOCAL_VALIDATION_TIMINGS']=str(work/'component_times_reordered.json')
    with (work/'reordered.log').open('w') as log:subprocess.run([sys.executable,'-c',script],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    reordered=pd.read_csv(work/'submission.csv',keep_default_na=False)
    assert dict(zip(output.audio_filename,output.transcript))==dict(zip(reordered.audio_filename,reordered.transcript))
    assert not (work/'reordered.log').read_text().strip()
    assert digest(archive)==inventory['sha256']
    for name,r in inventory['entries'].items():assert digest(source/name)==r['sha256']
    result={'zip':str(archive.resolve()),'sha256':inventory['sha256'],'zip_crc':True,'exact_archive_byte_content_verified':True,'complete_model_shard_maps':True,'offline':True,'empty_hf_cache':True,'socket_connections_blocked':True,'peft_and_sklearn_not_required':True,'normal_clip':True,'native_long_seconds':float(clips.iloc[1].duration_s),'long_tail_matched_reference_words':overlap,'csv_schema':True,'csv_quoting':True,'row_order':True,'no_nan':True,'no_blank_failure_rows':True,'repeat_runs':repeat,'deterministic_repeated_inference':repeat>1,'clip_independent_under_reordering':True,'no_transcript_or_test_logging':True,'inference_seconds':times,'reordered_inference_seconds':time.monotonic()-started,'device':'mps','official_a100_tested':False,'python':sys.version,'pass':True}
    result['local_only_component_timings']=[json.loads((work/f'component_times_{i+1}.json').read_text()) for i in range(repeat)]
    (work/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--repeat',type=int,default=2);a=p.parse_args();validate(a.zip.resolve(),a.label,a.repeat)
