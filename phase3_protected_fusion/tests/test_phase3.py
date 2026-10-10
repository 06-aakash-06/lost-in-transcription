import unittest,tempfile,sys,math,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'phase3_protected_fusion'))
import numpy as np,pandas as pd,soundfile as sf
from phase3_protected_fusion.runtime.fusion import fuse,alignment,blocks,repetition,collapse_single_runs
from phase3_protected_fusion.runtime.audio import load_audio
from phase3_protected_fusion.runtime.confidence import token_to_words
from phase3_protected_fusion.main import validate_output,select_device
from unittest.mock import patch
CONFIG={'mode':'confidence','threshold':.65,'max_span':2,'context':1,'collapse_anchor':True}
class Tests(unittest.TestCase):
    def test_confident_anchor_preserved(self):
        self.assertEqual(fuse('Ini Budi pulang.','Ini Budhi pulang.',CONFIG,{'anchor':[0,-.1,0]}),'Ini Budi pulang.')
    def test_low_confidence_local_correction_keeps_surface(self):
        self.assertEqual(fuse('Ini Budi pulang.','ini Budhi pulang',CONFIG,{'anchor':[0,-2,0]}),'Ini Budhi pulang.')
    def test_long_or_boundary_disagreement_retains_anchor(self):
        a='Ini satu dua tiga empat pulang.';b='Ini five six seven eight pulang.'
        self.assertEqual(fuse(a,b,CONFIG,{'anchor':[-3]*6}),a)
        self.assertEqual(fuse('Budi pulang.','Budhi pulang.',CONFIG,{'anchor':[-3]*2}),'Budi pulang.')
    def test_insert_delete_disabled(self):
        self.assertEqual(fuse('Ini Budi pulang.','Ini Pak Budi pulang.',CONFIG,{'anchor':[-3]*3}),'Ini Budi pulang.')
    def test_missing_long_clip_confidence_retains_anchor(self):
        self.assertEqual(fuse('Ini Budi pulang.','Ini Budhi pulang.',CONFIG,{'anchor':[None]*3}),'Ini Budi pulang.')
    def test_repetition_loop_not_ordinary_repetition(self):
        self.assertEqual(collapse_single_runs('iya iya iya'),'iya iya iya')
        self.assertEqual(fuse('iya '*6,'tidak',CONFIG),'iya iya')
        self.assertGreater(repetition('satu dua '*10),.9)
        self.assertEqual(fuse('Ini Budi pulang.','Ini '+('Budhi '*20)+'pulang.',CONFIG,{'anchor':[-3]*3}),'Ini Budi pulang.')
    def test_alignment_edges_deterministic(self):
        self.assertEqual(alignment([],['x']),[(None,0,False)])
        self.assertEqual(alignment(['x'],[]),[(0,None,False)])
        self.assertEqual(blocks(['Ini','Budi','pulang'],['ini','Budhi','pulang'])[0].start,1)
        results=[fuse('Ini Budi pulang.','Ini Budhi pulang.',CONFIG,{'anchor':[0,-2,0]}) for _ in range(10)]
        self.assertEqual(len(set(results)),1)
    def test_third_support(self):
        cfg={'mode':'support','max_span':3,'context':1}
        self.assertEqual(fuse('Ini Budi pulang.','Ini Budhi pulang.',cfg,third='Ini Budhi pulang.'),'Ini Budhi pulang.')
        self.assertEqual(fuse('Ini Budi pulang.','Ini Budhi pulang.',cfg,third='Ini Budi pulang.'),'Ini Budi pulang.')
        hybrid={**CONFIG,'mode':'hybrid'}
        self.assertEqual(fuse('Ini Budi pulang.','Ini Budhi pulang.',hybrid,{'anchor':[0,-.1,0]},third='Ini Budhi pulang.'),'Ini Budhi pulang.')
    def test_word_geometric_confidence_offsets(self):
        self.assertEqual(token_to_words('dua kata',[(0,0),(0,2),(2,3),(4,8)],[-99,-1,-3,-.5]),[-2,-.5])
    def test_strict_substitutions_cannot_change_word_count(self):
        cfg={'mode':'support','max_span':3,'context':1,'operations':['substitution'],'strict_substitutions':True}
        a='Ini Pak Budi pulang.';b='Ini Budhi pulang.'
        self.assertEqual(fuse(a,b,cfg,third=b),a)
        self.assertEqual(fuse('Ini Budi pulang.',b,cfg,third=b),b)
    def test_agreement_and_low_confidence_both_required(self):
        cfg={'mode':'support_confidence','threshold':.65,'max_span':3,'context':1,'operations':['substitution','insertion','deletion'],'allow_deletion':True}
        a='Ini Budi pulang.';b='Ini Budhi pulang.'
        self.assertEqual(fuse(a,b,cfg,{'anchor':[0,-2,0]},b),b)
        self.assertEqual(fuse(a,b,cfg,{'anchor':[0,-.1,0]},b),a)
        self.assertEqual(fuse(a,b,cfg,{'anchor':[None]*3},b),a)
        self.assertEqual(fuse(a,b,cfg,{'anchor':[0,-2,0]},a),a)
    def test_cuda_first_selection(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.backends.mps.is_available',side_effect=AssertionError('MPS queried before CUDA')):
            self.assertEqual(select_device(),'cuda')
        with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(select_device(),'mps')
    def test_audio_stereo_resampling_float(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.wav';w=np.ones((8000,2),dtype=np.float32)*[.2,.4];sf.write(p,w,8000,subtype='FLOAT')
            x=load_audio(p);self.assertEqual(x.dtype,np.float32);self.assertEqual(x.ndim,1);self.assertEqual(len(x),16000);self.assertAlmostEqual(float(x[8000]),.3,places=4)
    def test_csv_quoting_order_and_no_nan(self):
        meta=pd.DataFrame({'audio_filename':['b,".wav','a.wav']});out=pd.DataFrame({'audio_filename':meta.audio_filename,'transcript':['Hello, "world"','second transcript']})
        validate_output(meta,out)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.csv';out.to_csv(p,index=False);read=pd.read_csv(p,keep_default_na=False);self.assertEqual(out.to_dict(),read.to_dict())
        for broken in [out.iloc[::-1],out.iloc[:1],out.assign(transcript=[None,'x']),out.assign(transcript=['','x']),out.assign(transcript=['   ','x']),out.rename(columns={'transcript':'text'})]:
            with self.assertRaises(ValueError):validate_output(meta,broken)
if __name__=='__main__':unittest.main()
