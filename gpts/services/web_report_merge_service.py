import copy
from typing import Any, Dict, List


class WebReportMergeService:
    def build_effective_result(
        self,
        session_id: str,
        ai_result: Dict[str, Any],
        assets: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = copy.deepcopy(ai_result or {})
        teeth = merged.get("teeth", []) or []
        override_teeth = (overrides or {}).get("teeth", {}) or {}
        teeth_by_label: Dict[str, Dict[str, Any]] = {}

        for tooth in teeth:
            label = str(tooth.get("tooth_label", ""))
            if label:
                teeth_by_label[label] = tooth
            if not label or label not in override_teeth:
                tooth.setdefault("included", True)
                continue
            tooth_override = override_teeth[label]
            tooth.setdefault("included", True)
            for field in ("caries", "periapical", "missing", "implant", "crown", "filling", "bone_loss_level", "bone_loss_pct", "note", "included"):
                if field in tooth_override:
                    tooth[field] = tooth_override[field]

        pbl_map = {}
        pbl_level_map = {}
        bonelevel = {}
        caries_list: List[Dict[str, Any]] = []
        periapical_list: List[Dict[str, Any]] = []
        missing_list: List[str] = []
        missing_objects = copy.deepcopy((ai_result or {}).get("missing_teeth") or [])
        filtered_missing_objects: List[Any] = []
        implant_map: Dict[str, Dict[str, Any]] = {}
        crown_map: Dict[str, Dict[str, Any]] = {}
        filling_map: Dict[str, Dict[str, Any]] = {}
        caries_best_map: Dict[str, Dict[str, Any]] = {}
        periapical_best_map: Dict[str, Dict[str, Any]] = {}

        report_teeth: List[Dict[str, Any]] = []

        for tooth in teeth:
            label = str(tooth.get("tooth_label", ""))
            if not label:
                continue
            if tooth.get("included", True) is False:
                continue

            report_teeth.append(tooth)

            pbl_pct = float(tooth.get("bone_loss_pct", 0) or 0)
            pbl_level = int(tooth.get("bone_loss_level", 0) or 0)
            pbl_map[label] = pbl_pct
            pbl_level_map[label] = pbl_level
            bonelevel[label] = {"percent": pbl_pct, "level": pbl_level}

            if tooth.get("caries"):
                caries_list.append({"tooth_label": label})
                caries_best_map[label] = self._pick_best_finding(tooth, "caries")
            if tooth.get("periapical"):
                periapical_list.append({"tooth_label": label})
                periapical_best_map[label] = self._pick_best_finding(tooth, "periapical")
            if tooth.get("missing"):
                missing_list.append(label)
            if tooth.get("implant"):
                implant_map[label] = {"conf": 1.0, "box": tooth.get("box", [])}
            if tooth.get("crown"):
                crown_map[label] = {"conf": 1.0, "box": tooth.get("box", [])}
            if tooth.get("filling"):
                filling_map[label] = {"conf": 1.0, "box": tooth.get("box", [])}

        for missing_item in missing_objects:
            if isinstance(missing_item, dict):
                label = str(missing_item.get("tooth_label", ""))
                if not label:
                    continue
                linked_tooth = teeth_by_label.get(label)
                if linked_tooth is not None:
                    missing_item["included"] = linked_tooth.get("included", True)
                    missing_item["missing"] = linked_tooth.get("missing", True)
                    missing_item["note"] = linked_tooth.get("note", missing_item.get("note", ""))
                    if linked_tooth.get("included", True) is False or not linked_tooth.get("missing"):
                        continue
                filtered_missing_objects.append(missing_item)
                continue

            label = str(missing_item)
            linked_tooth = teeth_by_label.get(label)
            if linked_tooth is not None and (
                linked_tooth.get("included", True) is False or not linked_tooth.get("missing")
            ):
                continue
            filtered_missing_objects.append(label)

        merged["data"] = teeth
        merged["teeth"] = teeth
        merged["report_teeth"] = report_teeth
        merged["pbl"] = pbl_map
        merged["pbl_level"] = pbl_level_map
        merged["bonelevel"] = bonelevel
        merged["caries"] = caries_list
        merged["periapical"] = periapical_list
        merged["missing_teeth"] = filtered_missing_objects if missing_objects else missing_list
        merged["teeth_missing"] = missing_list
        merged["caries_by_tooth"] = caries_best_map
        merged["caries_by_tooth_best"] = caries_best_map
        merged["periapical_by_tooth"] = periapical_best_map
        merged["periapical_by_tooth_best"] = periapical_best_map
        merged["implant_by_tooth"] = implant_map
        merged["implant_by_tooth_best"] = implant_map
        merged["crown_by_tooth"] = crown_map
        merged["crown_by_tooth_best"] = crown_map
        merged["filling_by_tooth"] = filling_map
        merged["filling_by_tooth_best"] = filling_map
        merged["det_counts"] = {
            "seg_teeth": len(teeth),
            "caries": len(caries_list),
            "periapical": len(periapical_list),
            "cej_masks": int((ai_result or {}).get("det_counts", {}).get("cej_masks", 0)),
            "bonelevel_masks": int((ai_result or {}).get("det_counts", {}).get("bonelevel_masks", 0)),
        }
        merged["image_url"] = self._public_url(session_id, assets.get("source_path"))
        merged["preview_url"] = self._public_url(session_id, assets.get("preview_path"))
        merged["overlay_url"] = self._public_url(session_id, assets.get("overlay_path"))
        merged["bl_viz_url"] = self._public_url(session_id, assets.get("bl_viz_path"))
        merged["report_note"] = (overrides or {}).get("report_note", "")
        merged["attached_captures"] = copy.deepcopy((overrides or {}).get("attached_captures") or [])
        return merged

    def build_overrides_from_snapshot(
        self,
        ai_result: Dict[str, Any],
        snapshot: Dict[str, Any],
        current_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        restored: Dict[str, Any] = {
            "teeth": {},
            "report_note": (snapshot or {}).get("report_note", ""),
            "attached_captures": copy.deepcopy((snapshot or {}).get("attached_captures") or []),
        }

        ai_teeth = {
            str(tooth.get("tooth_label", "")): tooth
            for tooth in ((ai_result or {}).get("teeth") or [])
            if tooth.get("tooth_label")
        }
        snapshot_teeth = {
            str(tooth.get("tooth_label", "")): tooth
            for tooth in ((snapshot or {}).get("teeth") or [])
            if tooth.get("tooth_label")
        }
        report_teeth = {
            str(tooth.get("tooth_label", "")): tooth
            for tooth in ((snapshot or {}).get("report_teeth") or (snapshot or {}).get("teeth") or [])
            if tooth.get("tooth_label")
        }

        for tooth_label, ai_tooth in ai_teeth.items():
            snapshot_tooth = snapshot_teeth.get(tooth_label)
            if snapshot_tooth is None:
                continue

            tooth_override: Dict[str, Any] = {}
            for field in ("caries", "periapical", "missing", "implant", "crown", "filling", "bone_loss_level", "bone_loss_pct", "note"):
                snapshot_value = snapshot_tooth.get(field)
                ai_value = ai_tooth.get(field)
                if snapshot_value != ai_value:
                    tooth_override[field] = snapshot_value

            if tooth_label not in report_teeth:
                tooth_override["included"] = False
            elif ai_tooth.get("included", True) is False or snapshot_tooth.get("included", True) is not True:
                tooth_override["included"] = bool(snapshot_tooth.get("included", True))

            if tooth_override:
                restored["teeth"][tooth_label] = tooth_override

        return restored

    def _pick_best_finding(self, tooth: Dict[str, Any], finding_type: str) -> Dict[str, Any]:
        best = {"conf": 1.0, "box": tooth.get("box", [])}
        for finding in tooth.get("findings", []) or []:
            if str(finding.get("type", "")).lower() != finding_type:
                continue
            conf = float(finding.get("conf", finding.get("confidence", 0)) or 0)
            if conf >= float(best.get("conf", 0)):
                best = {"conf": conf, "box": finding.get("box", tooth.get("box", []))}
        return best

    def _public_url(self, session_id: str, path: Any) -> str | None:
        if not path:
            return None
        normalized = str(path).replace("\\", "/")
        marker = f"/{session_id}/"
        idx = normalized.rfind(marker)
        if idx >= 0:
            relative = normalized[idx + len(marker):]
        else:
            parts = normalized.split("/")
            try:
                sid_idx = parts.index(session_id)
                relative = "/".join(parts[sid_idx + 1 :])
            except ValueError:
                relative = parts[-1]
        return f"/api/web_report/session/{session_id}/files/{relative}"
