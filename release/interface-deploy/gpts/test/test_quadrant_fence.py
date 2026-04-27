import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.pano_rules_engine import RulesEngine


def _natural_obj(obj_id, fdi, cx, cy=370.0, conf=0.95):
    return {
        "id": obj_id,
        "type": "natural",
        "box": [cx - 20, cy - 50, cx + 20, cy + 50],
        "cx": cx,
        "cy": cy,
        "label_hint": str(fdi),
        "conf": conf,
        "contour": [],
    }


def test_q3_label_on_q4_side_is_not_allowed_to_keep_q3_slot():
    engine = RulesEngine(img_width=800, img_height=600)

    result = engine.run([_natural_obj(1, 37, cx=100.0)], [])
    slots = result["slots"]

    assert slots["37"]["object_id"] is None
    assert any(
        slots[str(fdi)]["object_id"] == 1
        for fdi in [48, 47, 46, 45, 44, 43, 42, 41]
    )


def test_q3_label_on_q3_side_can_keep_q3_slot():
    engine = RulesEngine(img_width=800, img_height=600)

    result = engine.run([_natural_obj(1, 37, cx=700.0)], [])

    assert result["slots"]["37"]["object_id"] == 1
