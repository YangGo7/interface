import sqlite3
import datetime
import os
from pathlib import Path
from collections import Counter

class StatsService:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to 'backend/data/stats.db'
            base_dir = Path(__file__).resolve().parent.parent
            data_dir = base_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "stats.db"
        else:
            self.db_path = db_path
            
        self._init_db()

    def _init_db(self):
        """Initialize stats database tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Visits Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # Findings Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                tooth_number INTEGER,
                finding_type TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_visit(self, session_id):
        """Log a page visit."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('INSERT INTO visits (session_id) VALUES (?)', (session_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[STATS] Log Visit Failed: {e}")

    def log_inference(self, session_id, model_findings):
        """
        Log inference results.
        model_findings: dict with 'problem_teeth' list.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            problem_teeth = model_findings.get('problem_teeth', [])
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for pt in problem_teeth:
                # pt = {'tooth': '16', 'issues': ['충치', '상악동 근접']}
                tooth_str = str(pt.get('tooth'))
                if not tooth_str.isdigit(): continue
                tooth_num = int(tooth_str)
                
                for issue in pt.get('issues', []):
                    # Simplify issue type for stats
                    ftype = 'other'
                    if '충치' in issue or 'caries' in issue.lower(): ftype = 'caries'
                    elif '치근단' in issue or 'periapical' in issue.lower(): ftype = 'periapical'
                    elif '치조골' in issue or 'bone' in issue.lower(): ftype = 'bone_loss'
                    elif '상악동' in issue or 'sinus' in issue.lower(): ftype = 'sinus'
                    elif '신경' in issue or 'nerve' in issue.lower(): ftype = 'nerve'
                    elif '임플란트' in issue or 'implant' in issue.lower(): ftype = 'implant'
                    
                    c.execute('''
                        INSERT INTO findings (timestamp, session_id, tooth_number, finding_type)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, session_id, tooth_num, ftype))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[STATS] Log Inference Failed: {e}")

    def get_dashboard_data(self):
        """Aggregate data for dashboard."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # 1. Visits (Last 7 Days)
            c.execute('''
                SELECT date(timestamp) as day, count(*) as count 
                FROM visits 
                WHERE timestamp >= date('now', '-7 days')
                GROUP BY date(timestamp)
                ORDER BY day ASC
            ''')
            visits_data = {row['day']: row['count'] for row in c.fetchall()}
            
            # Fill missing days with 0
            today = datetime.date.today()
            labels = []
            counts = []
            for i in range(6, -1, -1):
                day = (today - datetime.timedelta(days=i)).isoformat()
                labels.append(day)
                counts.append(visits_data.get(day, 0))
            
            visit_chart = {
                "labels": labels,
                "data": counts
            }

            # 2. Finding Types Ratio (All time)
            c.execute('''
                SELECT finding_type, count(*) as count 
                FROM findings 
                GROUP BY finding_type
            ''')
            type_counts = c.fetchall()
            type_chart = {
                "labels": [row['finding_type'] for row in type_counts],
                "data": [row['count'] for row in type_counts]
            }
            
            # 3. Top Problematic Teeth (Top 10)
            c.execute('''
                SELECT tooth_number, count(*) as count 
                FROM findings 
                WHERE finding_type IN ('caries', 'periapical', 'bone_loss')
                GROUP BY tooth_number
                ORDER BY count DESC
                LIMIT 10
            ''')
            tooth_counts = c.fetchall()
            tooth_chart = {
                "labels": [str(row['tooth_number']) for row in tooth_counts],
                "data": [row['count'] for row in tooth_counts]
            }
            
            conn.close()
            
            return {
                "visits": visit_chart,
                "types": type_chart,
                "teeth": tooth_chart
            }
        except Exception as e:
            print(f"[STATS] Get Data Failed: {e}")
            return {}
