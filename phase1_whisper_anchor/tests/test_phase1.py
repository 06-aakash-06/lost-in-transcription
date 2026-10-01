from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from inference.audio import RATE, load_audio, windows
from inference.postprocess import clean, merge_overlap, catastrophic_repeat, collapse_catastrophic
from training.text import to_reference_style
from training.metrics import corpus
from submission.main import validate_output
from submission.build_submission import build

class Phase1Tests(unittest.TestCase):
    def test_audio(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"stereo.wav"
            wave=np.stack([np.ones(8000,dtype=np.float32)*.4,np.zeros(8000,dtype=np.float32)],axis=1)
            sf.write(path,wave,8000)
            out=load_audio(path)
            self.assertEqual(out.dtype,np.float32)
            self.assertEqual(out.ndim,1)
            self.assertEqual(len(out),RATE)
            self.assertAlmostEqual(float(out.mean()),.2,places=2)
    def test_long_windows_cover_tail(self):
        wave=np.arange(40*RATE,dtype=np.float32)
        chunks=windows(wave)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks[0][0],0)
        self.assertEqual(chunks[-1][-1],wave[-1])
        self.assertEqual(chunks[0][-2*RATE],chunks[1][0])
    def test_long_training_split_preserves_audio_and_words(self):
        from training.train_whisper_lora import split_long_training
        wave=np.ones(36*RATE,dtype=np.float32)
        (left,lt),(right,rt)=split_long_training(wave,"satu dua tiga empat lima enam")
        self.assertEqual(len(left)+len(right),len(wave))
        self.assertLess(max(len(left),len(right)),30*RATE)
        self.assertEqual((lt+" "+rt).split(),"satu dua tiga empat lima enam".split())
    def test_text(self):
        self.assertEqual(to_reference_style("  Mba  I  ngomong…  gak-gak ’kan  "),"Mba I ngomong... gak-gak 'kan")
        self.assertEqual(clean("  siji  loro\n"),"siji loro")
        self.assertEqual(merge_overlap(["aku lagi sekolah", "lagi sekolah ning kene"]),"aku lagi sekolah ning kene")
        self.assertTrue(catastrophic_repeat("iki iki iki iki iki"))
        self.assertFalse(catastrophic_repeat("iki iki iki"))
        self.assertEqual(collapse_catastrophic("iki iki iki iki iki liyane"),"iki iki liyane")
    def test_official_score(self):
        value=corpus(["Halo, Dunia!", "Aku gak-gak"],["halo dunia", "aku gak gak"])
        self.assertEqual(value["words"],4)
        self.assertEqual(value["wer"],.75)  # Mid-sentence 'Dunia' remains capitalized.
    def test_submission_validation_and_csv(self):
        meta=pd.DataFrame({"audio_filename":["a.mp3","b.mp3"]})
        out=pd.DataFrame({"audio_filename":["a.mp3","b.mp3"],"transcript":["halo, dunia","baris\nbaru"]})
        validate_output(meta,out)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"submission.csv";out.to_csv(path,index=False)
            loaded=pd.read_csv(path,keep_default_na=False)
            self.assertEqual(loaded.transcript.tolist(),out.transcript.tolist())
        with self.assertRaises(ValueError): validate_output(meta,out.iloc[:1])
        with self.assertRaises(ValueError): validate_output(meta,out.assign(transcript=["ok",float("nan")]))
        with self.assertRaises(ValueError): validate_output(meta,out.iloc[::-1].reset_index(drop=True))
    def test_merged_model_offline_loading(self):
        model=HERE/"models/final_merged"
        if not model.exists(): self.skipTest("merged LoRA model has not been trained")
        import os
        from inference.whisper import WhisperAnchor
        old_hub,old_transformers=os.environ.get("HF_HUB_OFFLINE"),os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        try: self.assertIsNotNone(WhisperAnchor(model,"indonesian",1).model)
        finally:
            if old_hub is None: os.environ.pop("HF_HUB_OFFLINE",None)
            else: os.environ["HF_HUB_OFFLINE"]=old_hub
            if old_transformers is None: os.environ.pop("TRANSFORMERS_OFFLINE",None)
            else: os.environ["TRANSFORMERS_OFFLINE"]=old_transformers
    def test_split_design(self):
        a=pd.read_csv(HERE/"data/fold_A_train.tsv",sep="\t")
        av=pd.read_csv(HERE/"data/fold_A_valid.tsv",sep="\t")
        b=pd.read_csv(HERE/"data/fold_B_train.tsv",sep="\t")
        bv=pd.read_csv(HERE/"data/fold_B_valid.tsv",sep="\t")
        held=pd.read_csv(HERE/"data/jember_holdout.tsv",sep="\t")
        self.assertEqual((len(a),len(av),len(b),len(bv),len(held)),(1387,294,1603,78,162))
        self.assertEqual(set(a[a.domain=="competition"].convo_id),{2.0})
        self.assertEqual(set(av.convo_id),{5})
        self.assertEqual(set(b[b.domain=="competition"].convo_id),{5.0})
        self.assertEqual(set(bv.convo_id),{2})
        self.assertTrue(set(a[a.domain=="jember"].session).isdisjoint(held.session))
    def test_deterministic_zip_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/"model";model.mkdir()
            for name in ("config.json","generation_config.json","preprocessor_config.json","tokenizer_config.json","model.safetensors"):
                (model/name).write_text("fixture")
            first=root/"first.zip";second=root/"second.zip"
            self.assertEqual(build(model,first,"indonesian"),build(model,second,"indonesian"))
            import zipfile
            with zipfile.ZipFile(first) as archive:
                self.assertIn("main.py",archive.namelist())
                self.assertIn("model/model.safetensors",archive.namelist())
                self.assertNotIn("training/train_whisper_lora.py",archive.namelist())
    def test_runner_preserves_manifest_order(self):
        import json
        from unittest.mock import patch
        from submission.main import run
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/"clips").mkdir();(base/"model").mkdir()
            for name in ("b.wav","a.wav"):
                sf.write(base/"clips"/name,np.zeros(RATE,dtype=np.float32),RATE)
            pd.DataFrame({"audio_filename":["b.wav","a.wav"],"language":["ind","jav"]}).to_csv(base/"test_metadata.csv",index=False)
            (base/"config.json").write_text(json.dumps({"language":"indonesian","batch_size":2}))
            class FakeAnchor:
                def __init__(self,*args): pass
                def transcribe_arrays(self,waves): return ["halo, dunia","baris\nbaru"]
            with patch("submission.main.WhisperAnchor",FakeAnchor):
                run(base,base/"submission/submission.csv",base/"model",base/"config.json")
            result=pd.read_csv(base/"submission/submission.csv",keep_default_na=False)
            self.assertEqual(result.audio_filename.tolist(),["b.wav","a.wav"])
            self.assertEqual(result.transcript.tolist(),["halo, dunia","baris\nbaru"])

if __name__=="__main__": unittest.main()
