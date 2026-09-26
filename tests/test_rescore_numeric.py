import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_setup'))
from rescore_numeric import score


class NumericRescoreTests(unittest.TestCase):
    def test_heading_newlines_and_boxed(self):
        for text in ['Final Answer:\n$$\n\\boxed{30}\n$$','Final Answer:\n\n\\boxed{30}\nThere are 30 units.','Answer: 30']:
            self.assertTrue(score(text,'30')['numeric_correct'])
        self.assertEqual(score('Final Answer:\n$$\n\\boxed{30}\n$$','30')['extraction_method'],'boxed')
    def test_numeric_equivalence_and_wrong_answer(self):
        self.assertTrue(score('Answer: 1/2','0.5')['numeric_correct'])
        self.assertTrue(score('#### 1,200','1200')['numeric_correct'])
        self.assertFalse(score('Answer: 31','30')['numeric_correct'])
        self.assertFalse(score('No answer','30')['numeric_correct'])
