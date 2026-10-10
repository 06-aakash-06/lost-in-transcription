"""Emergency checkpoint hook for this already-running Python 3.14 trainer.

Installed with sys.remote_exec, then saves at the next completed microbatch.
It preserves uncommitted accumulated gradients and exits without another job.
"""
import sys
import os
import json
import time
import random
import datetime
from pathlib import Path
import numpy as np
import torch

def _save_and_stop(frame):
    sys.settrace(None); frame.f_trace = None
    values = frame.f_locals
    output = values['output']
    batch = values['batch_index'] + 1
    checkpoint = output / f'interrupted_epoch_{values["epoch"]+1}_batch_{batch:06d}'
    checkpoint.mkdir(exist_ok=False)
    state = {'config': values['config'], 'manifest': str(values['manifest'].resolve()),
        'optimizer': values['optimizer'].state_dict(), 'scheduler': values['scheduler'].state_dict(),
        'epoch': values['epoch'], 'next_batch': batch, 'indices': values['indices'],
        'sampler_rng': values['generator'].get_state(), 'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
        'device_rng': torch.mps.get_rng_state() if values['device']=='mps' else torch.cuda.get_rng_state_all() if values['device']=='cuda' else None,
        'step': values['step'], 'loss_sum': values['loss_sum'],
        'epoch_elapsed_s': time.monotonic()-values['started']+values['epoch_prior_elapsed'],
        'total_elapsed_s': time.monotonic()-values['run_started']+values['prior_elapsed'],
        'gradients': {name: p.grad.detach().cpu().clone() for name,p in values['model'].named_parameters() if p.requires_grad and p.grad is not None},
        'accumulated_microbatches': batch % values['accumulation'],
        'stop_reason': 'User requested immediate stop and preservation; no evaluation or next stage.'}
    values['model'].save_pretrained(checkpoint,safe_serialization=True)
    values['processor'].save_pretrained(checkpoint)
    temp=checkpoint/'training_state.tmp';torch.save(state,temp);temp.replace(checkpoint/'training_state.pt')
    metrics={'epoch':values['epoch']+1,'batch':batch,'batches_per_epoch':values['batches'],
        'optimizer_step':values['step'],'epoch_fraction':values['epoch']+batch/values['batches'],
        'mean_loss':values['loss_sum']/batch,'last_completed_batch_loss':values['raw_value'],
        'learning_rate':values['scheduler'].get_last_lr()[0],'accumulated_microbatches':state['accumulated_microbatches'],
        'gradient_tensors_saved':len(state['gradients']),'checkpoint':str(checkpoint.resolve()),
        'elapsed_s':state['epoch_elapsed_s'],'total_elapsed_s':state['total_elapsed_s'],
        'stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':state['stop_reason']}
    (checkpoint/'train_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    result=output.parent.parent/'results/stop_result.json'
    result.write_text(json.dumps(metrics,indent=2)+'\n')
    print('USER STOP: SAVED',json.dumps(metrics),flush=True)
    os._exit(0)

def _stop_trace(frame,event,arg):
    if frame.f_code.co_name=='train' and frame.f_code.co_filename.endswith('phase4_stronger_anchor/train.py'):
        # Line 114 in the code loaded by the running process is immediately
        # after backward/optional optimizer/scheduler/zero_grad/step increment.
        if event=='line' and frame.f_lineno==114:_save_and_stop(frame)
        return _stop_trace
    return None

_target=None
for _top in sys._current_frames().values():
    _frame=_top
    while _frame is not None:
        if _frame.f_code.co_name=='train' and _frame.f_code.co_filename.endswith('phase4_stronger_anchor/train.py'):
            _target=_frame;break
        _frame=_frame.f_back
    if _target is not None:break
if _target is None:raise RuntimeError('running trainer frame not found')
sys.settrace(_stop_trace);_target.f_trace=_stop_trace
print('USER STOP: checkpoint hook installed at batch',_target.f_locals.get('batch_index',-1)+1,flush=True)
