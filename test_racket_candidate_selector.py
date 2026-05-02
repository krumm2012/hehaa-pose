import unittest

from racket_candidate_selector import select_racket_candidate


class RacketCandidateSelectorTests(unittest.TestCase):
    def test_prefers_racket_near_active_ball_over_mirror_racket(self):
        rackets = [
            {"box": [1281, 90, 1373, 185], "confidence": 0.70},
            {"box": [1233, 460, 1397, 534], "confidence": 0.56},
        ]

        selected = select_racket_candidate(
            rackets,
            ball_position=[1229.0, 558.4],
            config={"frame_height": 1440},
        )

        self.assertEqual(selected["box"], [1233, 460, 1397, 534])

    def test_prefers_continuous_racket_when_confidence_is_close(self):
        rackets = [
            {"box": [1281, 90, 1373, 185], "confidence": 0.62},
            {"box": [1175, 468, 1424, 560], "confidence": 0.58},
        ]

        selected = select_racket_candidate(
            rackets,
            previous_center=[1315.0, 497.0],
            config={"frame_height": 1440},
        )

        self.assertEqual(selected["box"], [1175, 468, 1424, 560])


if __name__ == "__main__":
    unittest.main()
