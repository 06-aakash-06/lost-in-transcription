"""Meaningful invariants for the submission's conservative voting."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.runtime.consensus import consensus,unique_map

class ConsensusTests(unittest.TestCase):
    def test_two_supporters_required(self):
        self.assertEqual(consensus('aku mangan sega','aku mangan sego','aku mangan sega'),'aku mangan sega')
        self.assertEqual(consensus('aku mangan sega','aku mangan sego','aku mangan sego'),'aku mangan sego')
    def test_surface_agreement_required(self):
        self.assertEqual(consensus('aku mangan sega','aku mangan sego','aku mangan Sego'),'aku mangan sega')
    def test_no_insertions_deletions(self):
        self.assertEqual(consensus('aku mangan sega','aku wis mangan sega','aku wis mangan sega'),'aku mangan sega')
        self.assertEqual(consensus('aku wis mangan sega','aku mangan sega','aku mangan sega'),'aku wis mangan sega')
    def test_ambiguous_repeated_alignment_is_rejected(self):
        self.assertNotIn(0,unique_map(['aku','aku'],['aku']))
        self.assertNotIn(1,unique_map(['aku','aku'],['aku']))
    def test_repetition_safety(self):
        self.assertEqual(consensus('aku aku aku aku aku aku','aku','aku'),'aku aku')
    def test_missing_confidence_does_not_allow_rescue(self):
        self.assertEqual(consensus('aku mangan sega','aku mangan sego','aku mangan sego',threshold=.8),'aku mangan sega')
    def test_context_and_clip_independence(self):
        samples=[('sega mangan','sego mangan','sego mangan'),('aku mangan sega','aku mangan sego','aku mangan sego')]
        outputs=[consensus(*s) for s in samples]
        self.assertEqual(outputs,[consensus(*s) for s in reversed(samples)][::-1])
        self.assertEqual(consensus(*samples[0],context=1),'sega mangan')

if __name__=='__main__':unittest.main()
