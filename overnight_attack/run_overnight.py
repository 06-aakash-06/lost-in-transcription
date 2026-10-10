"""Execute gated model work with retries, fresh process residency, and reports."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'overnight_attack';OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from overnight_attack.evidence import fold_evidence,two_fold

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default

def checkpoint_latest(folder):
    candidates=[]
    for path in folder.glob('*/training_state.pt'):
        metrics=read(path.parent/'train_metrics.json',{})
        if metrics and (path.parent/'adapter_model.safetensors').exists():candidates.append((metrics['epoch_fraction'],path.parent))
    return max(candidates,key=lambda pair:pair[0])[1] if candidates else None

class Runner:
    def __init__(self,start,initial_pid):
        self.start=start;self.deadline=start+10*3600;self.initial_pid=initial_pid;self.state=read(OUT/'overnight_state.json',{}) or {}
        self.state.update(start_utc=dt.datetime.fromtimestamp(start,dt.timezone.utc).isoformat(),compute_deadline_utc=dt.datetime.fromtimestamp(self.deadline,dt.timezone.utc).isoformat())
    def hours(self):return max(0,(self.deadline-time.time())/3600)
    def record(self,stage,**data):
        self.state.update(stage=stage,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),remaining_compute_hours=self.hours(),**data)
        temp=OUT/'overnight_state.tmp';temp.write_text(json.dumps(self.state,indent=2)+'\n');temp.replace(OUT/'overnight_state.json')
        self.report();print(json.dumps({'stage':stage,**data}),flush=True)
    def report(self):
        recipe=read(OUT/'selected_large_recipe.json',{})
        large=read(OUT/'large_v3_two_fold.json',{})
        rows=['# Overnight attack','',f"Status: {self.state.get('stage','starting')}; job {self.state.get('job','')}. Updated UTC: {self.state.get('updated_utc','')}",'']
        progress=self.state.get('training_progress',{})
        if progress:rows.append(f"Latest training progress: epoch {progress.get('epoch')}, batch {progress.get('batch')}/{progress.get('batches_per_epoch')}, mean loss {progress.get('mean_loss',float('nan')):.6f}.")
        rows+=['## LARGE-V3',f"MPS: {self.state.get('large_mps','real BF16 smoke passed; full encoder+decoder r16, alpha32, dropout .05, Ratio-4, microbatch1/accum16')}"]
        if self.state.get('large_v3_outcome'):rows.append('Outcome: '+self.state['large_v3_outcome'])
        for fold in ['fold_A','fold_B']:
            evidence=read(OUT/f'large_v3_{fold}_evidence.json',{})
            for checkpoint,value in evidence.get('checkpoints',{}).items():
                rows.append(f"{fold} {checkpoint}: raw {value['raw']['wer']:.6f}, safe {value['safe']['wer']:.6f}, Jember {value.get('jember',{}).get('wer',float('nan')):.6f}; pair oracle Turbo {value['pair_oracle_Turbo_R3']['wer']:.6f}, MERaLiON {value['pair_oracle_MERaLiON']['wer']:.6f}. Gate {value['passes_continue_gate']}.")
        if large:
            best=large['candidates'][large['selected_checkpoint']]
            rows.append(f"Selected {large['selected_checkpoint']}: aggregate {best['safe']['wer']:.6f}; worst fold {best['worst_fold']:.6f}; Jember {best['jember']['wer']:.6f}; triple oracle {best['triple_oracle']['wer']:.6f}; final gate {large['passes_final_training_gate']}.")
        rows+=['Checkpoint paths: `overnight_attack/runs/large_v3_fold_A`, `large_v3_fold_B`.','', '## FINAL LARGE-V3',self.state.get('final_large_status','Not started; awaiting two-fold gate.')]
        final=HERE/'runs/final_large_v3';latest=checkpoint_latest(final) if final.exists() else None
        if latest:rows.append('Latest usable checkpoint: `'+str(latest)+'`.')
        rows+=['','## RATIO-4 FINAL',self.state.get('turbo_status','Historical interrupted checkpoint retained; fallback not started.')]
        turbo=HERE/'runs/final_turbo_ratio4';latest=checkpoint_latest(turbo) if turbo.exists() else None
        if latest:rows.append('Latest usable checkpoint: `'+str(latest)+'`.')
        for title,name in [('MODEL SOUP','soup_summary.json'),('ADAPTIVE DECODING','adaptive_summary.json'),('N-BEST','nbest_summary.json')]:
            value=read(OUT/name,{})
            rows+=['',f'## {title}',value.get('report','Not tested yet.')]
        rows+=['','## BEST NEW DIRECTION',self.state.get('best_direction','Gated Large-v3 LoRA capacity experiment.'),'','## NEXT ACTION',self.state.get('next_action','Complete Large-v3 Fold A, then apply the evidence gate before Fold B.'),'']
        if self.state.get('failures'):rows+=['Failures/retries: '+json.dumps(self.state['failures'])]
        rows+=['','Existing working ZIPs/final checkpoints are preserved. No submission ZIP is built by this runner. Competition metrics use held-out conversations; final all-data models only use Jember diagnostics.','']
        (HERE/'OVERNIGHT_REPORT.md').write_text('\n'.join(rows))
    def job(self,name,command,folder=None,expected=None,timeout=3*3600):
        if expected and expected.exists():return True
        for attempt in range(3):
            if self.hours()<.03:return False
            actual=list(map(str,command))
            if folder:
                latest=checkpoint_latest(folder)
                if latest:
                    if '--resume' in actual:
                        position=actual.index('--resume');actual[position+1]=str(latest)
                    else:actual+=['--resume',str(latest)]
            logfile=OUT/f'{name}_attempt{attempt+1}.log'
            with logfile.open('a') as log:
                process=subprocess.Popen([sys.executable,*actual],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
                self.record('running',job=name,attempt=attempt+1,pid=process.pid,log=str(logfile))
                begun=time.monotonic();timed_out=False
                while process.poll() is None:
                    if time.monotonic()-begun>timeout or time.time()>self.deadline:
                        process.send_signal(signal.SIGTERM);timed_out=True
                        try:process.wait(timeout=60)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        break
                    if folder:self.state['training_progress']=read(folder/'progress.json',{})
                    self.record('running',job=name,attempt=attempt+1,pid=process.pid,log=str(logfile))
                    time.sleep(30)
                code=process.wait()
            success=code==0 and (expected is None or expected.exists())
            if success:self.record('job_complete',job=name);return True
            self.state.setdefault('failures',[]).append({'job':name,'attempt':attempt+1,'exit_code':code,'timed_out':timed_out,'log':str(logfile)})
            self.record('retrying',job=name,attempt=attempt+1)
            if timed_out:break
            time.sleep(3)
        return False
    def train_large(self,fold,stop_after=None,final=False):
        folder=HERE/'runs'/('final_large_v3' if final else f'large_v3_{fold}')
        manifest=ROOT/'phase1_whisper_anchor/data'/('final_train.tsv' if final else f'{fold}_train.tsv')
        command=[HERE/'train_large.py','--config',HERE/'configs/large_v3_bf16_full.json','--manifest',manifest,'--output',folder]
        if stop_after:command+=['--stop-after',stop_after]
        command+=['--max-runtime',str(max(30,self.hours()*3600-60))]
        expected=folder/('epoch_1_half/training_state.pt' if stop_after else 'epoch_1_full/training_state.pt')
        return self.job('train_'+folder.name+('_half' if stop_after else '_full'),command,folder,expected,timeout=max(60,min(4*3600,self.hours()*3600)))
    def evaluate_large(self,fold,checkpoint,final=False):
        run='final_large_v3' if final else f'large_v3_{fold}'
        for name in (['jember_holdout'] if final else [f'{fold}_valid','jember_holdout']):
            output=OUT/'predictions'/run/checkpoint/f'{name}.json'
            command=[HERE/'evaluate.py','--base',HERE/'models/large_v3_base','--dtype','bfloat16','--checkpoint',HERE/'runs'/run/checkpoint,
                '--manifest',ROOT/f'phase1_whisper_anchor/data/{name}.tsv','--output',output,'--batch-size','4']
            if not self.job(f'eval_{run}_{checkpoint}_{name}',command,expected=output,timeout=45*60):return False
        if not final:fold_evidence(fold,run)
        self.report();return True
    def initial_half(self):
        expected=HERE/'runs/large_v3_fold_A/epoch_1_half/training_state.pt'
        while not expected.exists():
            try:os.kill(self.initial_pid,0)
            except ProcessLookupError:break
            if time.time()-self.start>3*3600:
                os.kill(self.initial_pid,signal.SIGTERM);break
            self.state['training_progress']=read(HERE/'runs/large_v3_fold_A/progress.json',{})
            self.record('running',job='initial_large_v3_fold_A_half',pid=self.initial_pid)
            time.sleep(30)
        # Allow any in-progress serialization/process exit before loading a
        # second model, keeping accelerator residency strictly sequential.
        exit_wait_started=time.monotonic()
        while True:
            try:os.kill(self.initial_pid,0)
            except ProcessLookupError:break
            if time.monotonic()-exit_wait_started>90:
                os.kill(self.initial_pid,signal.SIGKILL);break
            time.sleep(2)
        return expected.exists() or self.train_large('fold_A','half')
    def fallback(self,reason):
        self.record('fallback',large_v3_outcome=reason,best_direction='Final Ratio-4 Turbo plus validated weight averaging and adaptive decoding.',
            next_action='Resume the saved Ratio-4 optimizer state, then test checkpoint soup on both held-out conversations.')
        folder=HERE/'runs/final_turbo_ratio4'
        interrupted=ROOT/'phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218'
        base_command=[HERE/'train_turbo_resume.py','--config',ROOT/'phase4_stronger_anchor/configs/ratio4_two_epochs.json',
            '--manifest',ROOT/'phase1_whisper_anchor/data/final_train.tsv','--output',folder,'--resume',interrupted]
        # Obtain useful new final checkpoints first, while reserving time for
        # the requested high-value fallbacks rather than spending the entire
        # remaining window on an unfinished second epoch.
        for stop,expected in [('half','epoch_1_half'),('full','epoch_1_full')]:
            if self.hours()<1.1:break
            okay=self.job(f'turbo_ratio4_{stop}',base_command+['--stop-after',stop,'--max-runtime',str(max(30,(self.hours()-1.0)*3600))],folder,folder/expected/'training_state.pt')
            self.record('fallback',turbo_status=('Saved '+expected if okay else 'Partial/failure preserved; proceeding to other fallbacks.'))
            if not okay:break
        if self.hours()>.25:self.job('model_soup',[HERE/'fallback_experiments.py','--step','soup','--deadline',str(self.deadline)],timeout=max(60,self.hours()*3600))
        if self.hours()>.15:self.job('adaptive_decode',[HERE/'fallback_experiments.py','--step','adaptive','--deadline',str(self.deadline)],timeout=max(60,self.hours()*3600))
        if self.hours()>1.0:self.job('nbest_rescore',[HERE/'fallback_experiments.py','--step','nbest','--deadline',str(self.deadline)],timeout=min(3600,self.hours()*3600))
        if self.hours()>.1:
            self.record('fallback',turbo_status='Continuing Ratio-4 toward 1.5/2.0 epochs with remaining compute.')
            self.job('turbo_ratio4_remaining',base_command+['--max-runtime',str(max(30,self.hours()*3600-60))],folder,folder/'epoch_2_full/training_state.pt',timeout=max(60,self.hours()*3600))
        soup_result=read(OUT/'soup_summary.json',{})
        adaptive_result=read(OUT/'adaptive_summary.json',{})
        direction='Final Ratio-4 Turbo: '+str(checkpoint_latest(folder))
        if soup_result.get('kept') and soup_result.get('final_adapter'):direction='Validated Turbo LoRA delta soup: '+soup_result['final_adapter']
        elif adaptive_result.get('keep'):direction='Final Ratio-4 Turbo plus '+adaptive_result['best']['alternative']+' on acoustic-confidence-gated clips.'
        self.record('finished',best_direction=direction,next_action='Compare the best new final Ratio-4/soup checkpoint against the current anchor and rebuild the protected ensemble with remaining deadline time.')
    def run(self):
        self.record('running',large_mps='PASSED: BF16 full encoder+decoder r16; 32 real microbatches, 2 updates, finite gradients, ~6.0GB MPS memory.')
        if not self.initial_half():return self.fallback('Large-v3 Fold A half failed after retries.')
        if not self.evaluate_large('fold_A','epoch_1_half'):return self.fallback('Large-v3 half-checkpoint evaluation failed after retries.')
        if not self.train_large('fold_A'):return self.fallback('Large-v3 Fold A full failed after retries.')
        if not self.evaluate_large('fold_A','epoch_1_full'):return self.fallback('Large-v3 full-checkpoint evaluation failed after retries.')
        gate=read(OUT/'large_v3_fold_A_evidence.json',{})
        if not gate.get('passes_continue_gate'):return self.fallback('Large-v3 Fold A did not pass WER/oracle/OOD gate.')
        self.record('large_fold_A_passed',next_action='Complete Fold B with the identical Large-v3 recipe.')
        for checkpoint,stop in [('epoch_1_half','half'),('epoch_1_full',None)]:
            if not self.train_large('fold_B',stop):return self.fallback('Large-v3 Fold B training failed after retries.')
            if not self.evaluate_large('fold_B',checkpoint):return self.fallback('Large-v3 Fold B evaluation failed after retries.')
        combined=two_fold();self.report()
        if not combined['passes_final_training_gate']:return self.fallback('Large-v3 failed the same-checkpoint two-fold final-training gate.')
        self.record('large_two_fold_passed',final_large_status='Starting all-data training now.',best_direction='Large-v3 LoRA, '+combined['selected_checkpoint'],
            next_action='Finish the selected all-data Large-v3 checkpoint, then rebuild the ensemble around its measured complementarity.')
        if self.train_large(None,'half',True):
            self.record('final_large_half_ready',final_large_status='Usable 0.5-epoch all-data checkpoint saved; continuing toward 1.0.')
            if combined['selected_checkpoint']=='epoch_1_half' or self.hours()>2.2:
                self.evaluate_large(None,'epoch_1_half',True)
                self.job('merge_final_large_half',[HERE/'merge.py','--base-model',HERE/'models/large_v3_base',
                    '--adapter',HERE/'runs/final_large_v3/epoch_1_half','--output',HERE/'models/final_large_v3_half'],expected=HERE/'models/final_large_v3_half/merge_validation.json',timeout=20*60)
            else:
                self.record('final_large_half_ready',final_large_status='0.5-epoch adapter and resumable state saved; optional half diagnostic/merge deferred to prioritize the validated 1.0-epoch checkpoint.')
        if self.hours()>.15 and self.train_large(None,None,True):
            self.record('final_large_full_ready',final_large_status='All-data 1.0 epoch complete; half and full adapters usable.')
            self.evaluate_large(None,'epoch_1_full',True)
            self.job('merge_final_large_full',[HERE/'merge.py','--base-model',HERE/'models/large_v3_base',
                '--adapter',HERE/'runs/final_large_v3/epoch_1_full','--output',HERE/'models/final_large_v3_full'],expected=HERE/'models/final_large_v3_full/merge_validation.json',timeout=20*60)
        if self.hours()>.3:self.job('large_checkpoint_soup',[HERE/'fallback_experiments.py','--step','large_soup','--deadline',str(self.deadline)],timeout=max(60,self.hours()*3600))
        if not (HERE/'runs/final_large_v3/epoch_1_full/training_state.pt').exists():
            self.state['final_large_status']='Window ended with the half checkpoint and latest resumable partial state preserved; full epoch is pending.'
        self.record('finished',next_action='Select the validated Large-v3 checkpoint, evaluate protected ensemble corrections, and build/test one final submission in the remaining deadline window.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start-utc',required=True);p.add_argument('--initial-pid',type=int,required=True);a=p.parse_args()
    runner=Runner(dt.datetime.fromisoformat(a.start_utc).timestamp(),a.initial_pid)
    try:runner.run()
    except Exception as exc:
        runner.record('unexpected_error',error=repr(exc))
        if runner.hours()>.1:
            try:runner.fallback('Unexpected Large-v3 branch error: '+repr(exc))
            except Exception as second:runner.record('failed',error=repr(second))
