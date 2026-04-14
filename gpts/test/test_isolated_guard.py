import unittest

from utils.post_processing.isolated_guard import build_isolated_guard_keep_indices


class IsolatedGuardTests(unittest.TestCase):
    def test_keeps_high_conf_and_isolated_low_conf(self):
        boxes = [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ]
        confidences = [0.90, 0.22, 0.23, 0.19]

        keep_indices = build_isolated_guard_keep_indices(boxes, confidences)

        self.assertEqual(keep_indices, [0, 2])

    def test_keeps_low_conf_when_no_high_conf_exists(self):
        boxes = [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ]
        confidences = [0.21, 0.24]

        keep_indices = build_isolated_guard_keep_indices(boxes, confidences)

        self.assertEqual(keep_indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
