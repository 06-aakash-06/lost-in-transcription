"""Exercise the real discovered phrase-loop case through both exact ZIPs."""
import ast,json,os,sys,subprocess,shutil,time
from pathlib import Path
import pandas as pd
from runtime.repetition_guard import guard
from runtime.fusion import repetition
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results'
def check():
    parsed=ast.parse((HERE/'validate_zip.py').read_text())
    script=next(n.value.value for n in ast.walk(parsed) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets) and isinstance(n.value,ast.Constant))
    metadata=pd.read_csv(ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv',sep='\t',keep_default_na=False)
    row=metadata.set_index('clip_id').loc['jem_041_00357']
    cached=pd.read_csv(ROOT/'overnight_attack/results/predictions/final_large_v3/epoch_1_full/jember_holdout.csv',keep_default_na=False).set_index('clip_id').loc['jem_041_00357'].transcript
    expected=guard(cached);assert len(cached.split())-len(expected.split())==168
    data=OUT/'offline_zip_tests/phrase_guard_case/data';(data/'clips').mkdir(parents=True,exist_ok=True)
    filename='repetition-case'+Path(row.path).suffix;shutil.copyfile(ROOT/row.path,data/'clips'/filename)
    pd.DataFrame({'audio_filename':[filename]}).to_csv(data/'test_metadata.csv',index=False)
    results={}
    for label in ['recommended','backup']:
        source=OUT/f'offline_zip_tests/{label}/extracted';work=OUT/f'offline_zip_tests/phrase_guard_case/{label}';work.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(HF_HOME=str(work/'empty_hf_cache'),HF_HUB_CACHE=str(work/'empty_hf_cache/hub'),HUGGINGFACE_HUB_CACHE=str(work/'empty_hf_cache/hub'),HF_MODULES_CACHE=str(work/'empty_hf_cache/modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LIT_DATA_DIR=str(data),LIT_SUBMISSION_PATH=str(work/'submission.csv'),PYTHONPATH='',LIT_LOCAL_VALIDATION_TIMINGS=str(work/'component_times.json'))
        started=time.monotonic()
        if not (work/'submission.csv').exists():
            with (work/'inference.log').open('w') as log:subprocess.run([sys.executable,'-c',script],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        output=pd.read_csv(work/'submission.csv',keep_default_na=False)
        assert list(output.columns)==['audio_filename','transcript'] and output.audio_filename.tolist()==[filename]
        assert output.transcript.map(lambda s:isinstance(s,str) and bool(s.strip())).all()
        assert repetition(output.transcript.iloc[0])<.35
        assert not (work/'inference.log').read_text().strip()
        replay="""import os,runpy
from pathlib import Path
import runtime.whisper as wh
import runtime.meralion as me
class ReplayWhisper:
    def __init__(self,path,*args):self.name=Path(path).name
    def transcribe_arrays(self,waves):return [os.environ['LIT_RECORDED_LOOP'] if self.name=='large' else 'different alternative']*len(waves)
class ReplayMer:
    def __init__(self,*args):pass
    def transcribe_arrays(self,waves):return ['another alternative']*len(waves)
wh.Whisper=ReplayWhisper;me.Meralion=ReplayMer
runpy.run_path('main.py',run_name='__main__')
"""
        replay_env=env.copy();replay_env.update(LIT_RECORDED_LOOP=cached,LIT_SUBMISSION_PATH=str(work/'replayed.csv'))
        with (work/'replay.log').open('w') as log:subprocess.run([sys.executable,'-c',replay],cwd=source,env=replay_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        replayed=pd.read_csv(work/'replayed.csv',keep_default_na=False);assert replayed.transcript.tolist()==[expected]
        assert not (work/'replay.log').read_text().strip()
        results[label]={'actual_zip_entrypoint':True,'live_case_is_nonrepeating_speech':True,'recorded_loop_replay_matches_guard':True,'replay_method':'ASR outputs stubbed only at engine boundary; exact archived main.py and guard execute unchanged','loop_words_removed_in_replay':168,'offline':True,'silent_inference':True,'elapsed_seconds':time.monotonic()-started,'pass':True}
    (OUT/'guard_zip_smoke.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':check()
