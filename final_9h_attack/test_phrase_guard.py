import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_9h_attack.runtime.repetition_guard import guard

class GuardTests(unittest.TestCase):
    def test_proven_single_word_rule(self):self.assertEqual(guard('aku '*8),'aku aku')
    def test_clear_dominant_phrase(self):self.assertEqual(guard('wis rampung '+('ora ngerti '*15)),'wis rampung ora ngerti ora ngerti')
    def test_short_natural_repetition_is_preserved(self):
        text='aku ngerti '*5;self.assertEqual(guard(text),text.strip())
    def test_nondominant_run_is_preserved(self):
        text=('awal '*2)+('aku ngerti '*10)+' '.join('tembung'+str(i) for i in range(80));self.assertEqual(guard(text),text)
    def test_six_word_loop_and_surface_preservation(self):
        phrase='Aku ora ngerti karepe sapa saiki';self.assertEqual(guard((phrase+' ')*6),phrase+' '+phrase)

if __name__=='__main__':unittest.main()
