import sys
import unittest
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_setup'))
from adaptive_two_stage_policy import recent_budget


class AcceptanceControllerTest(unittest.TestCase):
    def budget(self, history):
        return recent_budget(history, 4, 1, .25, .60)

    def test_response_history_and_recovery(self):
        history = deque(maxlen=2)
        self.assertEqual(self.budget(history), (1, None))
        history.append((0, 1))
        self.assertEqual(self.budget(history), (1, 0.0))
        # A skipped cycle does not append a false rejection. Trial drafts after
        # cooldown can reopen the route, even after a failed history.
        history.append((1, 1))
        self.assertEqual(self.budget(history), (2, .5))
        history.append((2, 2))
        self.assertEqual(self.budget(history), (4, 1.0))
        self.assertEqual(self.budget(deque(maxlen=2)), (1, None))

    def test_token_weighted_acceptance_and_max_budget(self):
        # Count tokens, not the mean of per-cycle percentages (which is .5).
        self.assertEqual(self.budget([(1, 1), (0, 4)]), (1, .2))
        self.assertEqual(recent_budget([(1, 1)], 1, 1, .25, .60), (1, 1.0))


if __name__ == '__main__':
    unittest.main()
