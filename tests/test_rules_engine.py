
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from services.pano_rules_engine import RulesEngine

class TestRulesEngine(unittest.TestCase):
    def test_metrics_calculation(self):
        # Setup: Anchors at known positions
        # Image Widht 1000. 
        # Anchors: 46 (x=200), 48 (x=100) -> Lower Right (Image Left)
        # 36 (x=800), 38 (x=900) -> Lower Left (Image Right)
        
        objects = [
            {'id': 1, 'type': 'natural', 'label_hint': '46', 'cx': 200, 'cy': 600, 'box': [180, 580, 220, 620]}, # W=40
            {'id': 2, 'type': 'natural', 'label_hint': '48', 'cx': 100, 'cy': 600, 'box': [80, 580, 120, 620]},  # W=40
            {'id': 3, 'type': 'natural', 'label_hint': '36', 'cx': 800, 'cy': 600, 'box': [780, 580, 820, 620]},
            {'id': 4, 'type': 'natural', 'label_hint': '38', 'cx': 900, 'cy': 600, 'box': [880, 580, 920, 620]},
        ]
        
        engine = RulesEngine(1000, 1000)
        u, l, y_split = engine._split_arches(objects)
        self.assertEqual(len(l), 4)
        
        # Test Metrics
        metrics = engine._calculate_metrics(l, is_upper=False)
        # dx between 48(100) and 46(200)?
        # 48->n=8, 46->n=6. Diff is 2 slots. Dist is 100. dx ~ 50.
        self.assertAlmostEqual(metrics['dx'], 50.0, delta=10.0)
        
        # w_base ~ 40
        self.assertAlmostEqual(metrics['w_base'], 40.0, delta=10.0)

    def test_implant_assignment(self):
        # Scenario: 46(200), 48(100). Gap at 47(150 ideally).
        # Implant at 150.
        objects = [
            {'id': 'anc1', 'type': 'natural', 'label_hint': '46', 'cx': 200, 'cy': 600, 'box': [180, 580, 220, 620]}, 
            {'id': 'anc2', 'type': 'natural', 'label_hint': '48', 'cx': 100, 'cy': 600, 'box': [80, 580, 120, 620]},
            {'id': 'imp1', 'type': 'implant', 'cx': 150, 'cy': 600, 'box': [130, 580, 170, 620]}
        ]
        
        engine = RulesEngine(1000, 1000)
        result = engine.run(objects, [])
        slots = result['slots']
        
        # Check 46, 48 are confirmed anchors
        self.assertEqual(slots['46']['status'], 'confirmed')
        self.assertEqual(slots['48']['status'], 'confirmed')
        
        # Check 47 is confirmed implant
        # Note: 47 ideal_x should be around 150
        self.assertEqual(slots['47']['status'], 'confirmed')
        self.assertEqual(slots['47']['object_id'], 'imp1')

    def test_spatial_order_warning(self):
        # Scenario: 11(400), 12(500). 
        # y=200 (Upper). Add a dummy lower object at y=800 to force y_split ~ 500.
        
        objects = [
            {'id': 't11', 'type': 'natural', 'label_hint': '11', 'cx': 400, 'cy': 200, 'box': [380, 180, 420, 220]},
            {'id': 't12', 'type': 'natural', 'label_hint': '12', 'cx': 500, 'cy': 200, 'box': [480, 180, 520, 220]},
            {'id': 'dummy1', 'type': 'natural', 'label_hint': '41', 'cx': 500, 'cy': 800, 'box': [480, 780, 520, 820]},
            {'id': 'dummy2', 'type': 'natural', 'label_hint': '42', 'cx': 400, 'cy': 800, 'box': [380, 780, 420, 820]}
        ]
        engine = RulesEngine(1000, 1000)
        result = engine.run(objects, [])
        
        # 11 and 12 should be assigned to slots '11' and '12' because they are anchors (natural + label)
        # But wait, logic says "Anchors are assigned to their own slots".
        # So slot 11 gets t11, slot 12 gets t12.
        # Then verify order: 11(400), 12(500).
        # Expected Upper: 18..11 (Mid) .. 28.
        # 12 is Left of 11 in Sequence (Sequence Index: 12->6, 11->7).
        # Sorted X: t11(400), t12(500).
        # Check: 
        # i=0: t11(idx 7). next=t12(idx 6). 
        # 7 > 6 -> Violation.
        
        self.assertEqual(result['slots']['11']['status'], 'ambiguous')
        self.assertEqual(result['slots']['12']['status'], 'ambiguous')

if __name__ == '__main__':
    unittest.main()
