import unittest
import numpy as np
from services.pano_rules_engine import RulesEngine

class TestSpatialSwap(unittest.TestCase):
    def test_spatial_swap_47_48(self):
        """Test if 47 (Left) and 48 (Right) get swapped correctly."""
        # Setup: 
        # Physical Reality: Obj1 at X=100 (Left), Obj2 at X=200 (Right).
        # Incorrect Assignment: Obj1 -> 47, Obj2 -> 48.
        # Why incorrect? 
        # Lower Arch: 48 is Most Distal (Left in Image). 47 is Medial (Right of 48).
        # So X(48) should be < X(47).
        # Here X(47)=100 < X(48)=200. This logic (Left->47, Right->48) violates 48<47.
        
        engine = RulesEngine(1000, 1000)
        
        # Manually create slots with valid internal structure required by _enforce_spatial_order
        # Note: 'candidates' expects list of dicts.
        obj_left = {'id': 'obj_left', 'cx': 100, 'type': 'natural', 'label_hint': '47'} # Hint 47 but at 48 pos?
        obj_right = {'id': 'obj_right', 'cx': 200, 'type': 'natural', 'label_hint': '48'}
        
        slots = {
            '47': {
                'fdi': '47', 'status': 'confirmed', 'candidates': [obj_left], 'object_id': 'obj_left'
            },
            '48': {
                'fdi': '48', 'status': 'confirmed', 'candidates': [obj_right], 'object_id': 'obj_right'
            },
            '46': {'fdi': '46', 'status': 'unassigned', 'object_id': None, 'candidates': []} 
        }
        
        # Run enforcement
        corrected = engine._enforce_spatial_order(slots, is_upper=False)
        
        # Expectation:
        # 48 should now hold the Left object (obj_left, X=100)
        # 47 should now hold the Right object (obj_right, X=200)
        
        self.assertEqual(corrected['48']['object_id'], 'obj_left')
        self.assertEqual(corrected['47']['object_id'], 'obj_right')

if __name__ == '__main__':
    unittest.main()
