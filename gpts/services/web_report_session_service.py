import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class WebReportSessionService:
    def __init__(self, db_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else data_dir / "web_report.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS web_report_sessions (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                language TEXT NOT NULL DEFAULT 'English',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finalized_at TEXT,
                is_finalized INTEGER NOT NULL DEFAULT 0,
                current_report_version INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS web_report_assets (
                session_id TEXT PRIMARY KEY,
                source_path TEXT,
                preview_path TEXT,
                overlay_path TEXT,
                bl_viz_path TEXT,
                inference_dir TEXT,
                reports_dir TEXT,
                final_dir TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS web_report_ai_results (
                session_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS web_report_doctor_overrides (
                session_id TEXT PRIMARY KEY,
                override_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS web_report_report_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                html_path TEXT,
                pdf_path TEXT,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def create_session(self, language: str = "English") -> str:
        session_id = str(uuid.uuid4())
        now = utc_now()
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO web_report_sessions (
                id, status, language, created_at, updated_at, is_finalized, current_report_version
            ) VALUES (?, ?, ?, ?, ?, 0, 0)
            """,
            (session_id, "waiting", language, now, now),
        )
        cur.execute(
            """
            INSERT INTO web_report_doctor_overrides (session_id, override_json, updated_at)
            VALUES (?, ?, ?)
            """,
            (session_id, json.dumps({"teeth": {}, "report_note": "", "attached_captures": []}), now),
        )
        conn.commit()
        conn.close()
        return session_id

    def set_status(self, session_id: str, status: str, error: Optional[str] = None) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE web_report_sessions
            SET status = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error, utc_now(), session_id),
        )
        conn.commit()
        conn.close()

    def set_assets(self, session_id: str, assets: Dict[str, Optional[str]]) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO web_report_assets (
                session_id, source_path, preview_path, overlay_path, bl_viz_path,
                inference_dir, reports_dir, final_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                source_path = excluded.source_path,
                preview_path = excluded.preview_path,
                overlay_path = excluded.overlay_path,
                bl_viz_path = excluded.bl_viz_path,
                inference_dir = excluded.inference_dir,
                reports_dir = excluded.reports_dir,
                final_dir = excluded.final_dir
            """,
            (
                session_id,
                assets.get("source_path"),
                assets.get("preview_path"),
                assets.get("overlay_path"),
                assets.get("bl_viz_path"),
                assets.get("inference_dir"),
                assets.get("reports_dir"),
                assets.get("final_dir"),
            ),
        )
        cur.execute(
            "UPDATE web_report_sessions SET updated_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )
        conn.commit()
        conn.close()

    def save_ai_result(self, session_id: str, result: Dict[str, Any]) -> None:
        now = utc_now()
        payload = json.dumps(result, ensure_ascii=False)
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO web_report_ai_results (session_id, result_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (session_id, payload, now),
        )
        cur.execute(
            "UPDATE web_report_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        conn.close()

    def save_overrides(self, session_id: str, overrides: Dict[str, Any], updated_by: Optional[str] = None) -> None:
        now = utc_now()
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO web_report_doctor_overrides (session_id, override_json, updated_at, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                override_json = excluded.override_json,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (session_id, json.dumps(overrides, ensure_ascii=False), now, updated_by),
        )
        cur.execute(
            "UPDATE web_report_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        conn.close()

    def create_report_version(
        self,
        session_id: str,
        status: str,
        html_path: Optional[str],
        pdf_path: Optional[str],
        snapshot: Dict[str, Any],
    ) -> int:
        now = utc_now()
        conn = self._connect()
        cur = conn.cursor()
        row = cur.execute(
            "SELECT COALESCE(MAX(version), 0) AS max_version FROM web_report_report_versions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        version = int(row["max_version"]) + 1
        cur.execute(
            """
            INSERT INTO web_report_report_versions (
                session_id, version, status, html_path, pdf_path, snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                version,
                status,
                html_path,
                pdf_path,
                json.dumps(snapshot, ensure_ascii=False),
                now,
            ),
        )
        if status == "final":
            cur.execute(
                """
                UPDATE web_report_sessions
                SET current_report_version = ?, updated_at = ?, is_finalized = 1, finalized_at = ?
                WHERE id = ?
                """,
                (version, now, now, session_id),
            )
        else:
            cur.execute(
                """
                UPDATE web_report_sessions
                SET current_report_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (version, now, session_id),
            )
        conn.commit()
        conn.close()
        return version

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        session_row = cur.execute(
            "SELECT * FROM web_report_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            conn.close()
            return None

        assets_row = cur.execute(
            "SELECT * FROM web_report_assets WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        ai_row = cur.execute(
            "SELECT * FROM web_report_ai_results WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        override_row = cur.execute(
            "SELECT * FROM web_report_doctor_overrides WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        report_row = cur.execute(
            """
            SELECT * FROM web_report_report_versions
            WHERE session_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()

        return {
            "id": session_row["id"],
            "status": session_row["status"],
            "error": session_row["error"],
            "language": session_row["language"],
            "created_at": session_row["created_at"],
            "updated_at": session_row["updated_at"],
            "finalized_at": session_row["finalized_at"],
            "is_finalized": bool(session_row["is_finalized"]),
            "current_report_version": int(session_row["current_report_version"]),
            "assets": dict(assets_row) if assets_row else {},
            "ai_result": json.loads(ai_row["result_json"]) if ai_row else None,
            "doctor_overrides": json.loads(override_row["override_json"]) if override_row else {"teeth": {}, "report_note": "", "attached_captures": []},
            "report": (
                {
                    "version": report_row["version"],
                    "status": report_row["status"],
                    "html_path": report_row["html_path"],
                    "pdf_path": report_row["pdf_path"],
                    "created_at": report_row["created_at"],
                }
                if report_row
                else None
            ),
        }

    def list_report_versions(self, session_id: str) -> list[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT version, status, html_path, pdf_path, created_at
            FROM web_report_report_versions
            WHERE session_id = ?
            ORDER BY version DESC
            """,
            (session_id,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
