"""Device, schema, and memory-failure behavior of the shipped entrypoint."""
import sys,unittest,importlib.util
from pathlib import Path
from unittest.mock import patch
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('submission_entrypoint',HERE/'main.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from runtime.whisper import Whisper

class EntrypointTests(unittest.TestCase):
    def test_cuda_precedes_mps(self):
        with patch.object(entry.torch.cuda,'is_available',return_value=True),patch.object(entry.torch.backends.mps,'is_available',return_value=True):
            self.assertEqual(entry.select_device(),'cuda')
    def test_mps_then_cpu(self):
        with patch.object(entry.torch.cuda,'is_available',return_value=False),patch.object(entry.torch.backends.mps,'is_available',return_value=True):
            self.assertEqual(entry.select_device(),'mps')
        with patch.object(entry.torch.cuda,'is_available',return_value=False),patch.object(entry.torch.backends.mps,'is_available',return_value=False):
            self.assertEqual(entry.select_device(),'cpu')
    def test_order_and_failure_rows_are_rejected(self):
        metadata=pd.DataFrame({'audio_filename':['b.mp3','a.mp3']})
        for names,texts in [(['a.mp3','b.mp3'],['x','y']),(['b.mp3','a.mp3'],['x','']),(['b.mp3','a.mp3'],['x',None])]:
            with self.assertRaises(ValueError):entry.validate_output(metadata,pd.DataFrame({'audio_filename':names,'transcript':texts}))
        entry.validate_output(metadata,pd.DataFrame({'audio_filename':['b.mp3','a.mp3'],'transcript':['x','y']}))
    def test_memory_retry_preserves_batch_order(self):
        engine=Whisper.__new__(Whisper);engine.device='cpu'
        def decode(waves):
            if len(waves)>1:raise RuntimeError('out of memory')
            return [str(waves[0])]
        engine._decode=decode
        self.assertEqual(engine._safe([3,1,2]),['3','1','2'])
    def test_non_memory_failure_is_not_swallowed(self):
        engine=Whisper.__new__(Whisper);engine.device='cpu'
        engine._decode=lambda waves:(_ for _ in ()).throw(RuntimeError('invalid features'))
        with self.assertRaisesRegex(RuntimeError,'invalid features'):engine._safe([1,2])

if __name__=='__main__':unittest.main()
