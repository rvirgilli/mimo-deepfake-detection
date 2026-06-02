"""
Results database for experiment aggregation and export.

This module provides a SQLite-based database for storing and querying
experiment results, with export capabilities for paper writing.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .experiment import ExperimentManifest


class ResultsDB:
    """
    SQLite database for experiment results.

    Provides storage, querying, and export functionality for
    experiment manifests and metrics.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the results database.

        Args:
            db_path: Path to SQLite database file.
                    Defaults to experiments/results.db
        """
        if db_path is None:
            # Default to project root experiments/results.db
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / "experiments" / "results.db")

        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    git_hash TEXT,
                    git_dirty INTEGER,
                    git_branch TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    hostname TEXT,
                    status TEXT,
                    model_save_path TEXT,

                    -- Flattened config fields for easy querying
                    frontend TEXT,
                    frontend_freeze INTEGER,
                    lr REAL,
                    weight_decay REAL,
                    loss_weight_spoof REAL,
                    rawboost_algo INTEGER,
                    batch_size INTEGER,
                    num_epochs INTEGER,
                    seed INTEGER,

                    -- Metrics
                    best_eer REAL,
                    best_val_loss REAL,
                    final_train_loss REAL,
                    epochs_completed INTEGER,
                    best_epoch INTEGER,

                    -- Full data as JSON
                    config_json TEXT,
                    metrics_json TEXT,
                    gpu_info_json TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_frontend
                ON experiments(frontend)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_start_time
                ON experiments(start_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_best_eer
                ON experiments(best_eer)
            """)
            conn.commit()

    def add_experiment(self, manifest: ExperimentManifest) -> None:
        """
        Add or update an experiment in the database.

        Args:
            manifest: ExperimentManifest to store
        """
        config = manifest.config
        metrics = manifest.metrics

        # Extract flattened config values
        frontend = config.get("frontend", {}).get("name", "unknown")
        frontend_freeze = 1 if config.get("frontend", {}).get("freeze", True) else 0
        training = config.get("training", {})
        rawboost = config.get("rawboost", {})
        dataset = config.get("dataset", {})

        loss_weights = training.get("loss_weights", [0.1, 0.9])
        loss_weight_spoof = loss_weights[0] if isinstance(loss_weights, list) else 0.1

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, experiment_name, git_hash, git_dirty, git_branch,
                    start_time, end_time, duration_seconds, hostname, status,
                    model_save_path,
                    frontend, frontend_freeze, lr, weight_decay, loss_weight_spoof,
                    rawboost_algo, batch_size, num_epochs, seed,
                    best_eer, best_val_loss, final_train_loss, epochs_completed, best_epoch,
                    config_json, metrics_json, gpu_info_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                manifest.experiment_id,
                manifest.experiment_name,
                manifest.git_hash,
                1 if manifest.git_dirty else 0,
                manifest.git_branch,
                manifest.start_time,
                manifest.end_time,
                manifest.duration_seconds,
                manifest.hostname,
                manifest.status,
                manifest.model_save_path,
                frontend,
                frontend_freeze,
                training.get("lr", 0),
                training.get("weight_decay", 0),
                loss_weight_spoof,
                rawboost.get("algo", 0),
                dataset.get("batch_size", 0),
                training.get("num_epochs", 0),
                config.get("seed", 0),
                metrics.get("best_eer"),
                metrics.get("best_val_loss"),
                metrics.get("final_train_loss"),
                metrics.get("epochs_completed"),
                metrics.get("best_epoch"),
                json.dumps(config),
                json.dumps(metrics),
                json.dumps(manifest.gpu_info),
            ))
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get a single experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_all(
        self,
        frontend: Optional[str] = None,
        status: str = "completed",
        order_by: str = "start_time",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all experiments matching criteria.

        Args:
            frontend: Filter by frontend name
            status: Filter by status (default: completed)
            order_by: Column to order by
            limit: Maximum number of results

        Returns:
            List of experiment dicts
        """
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if frontend:
            query += " AND frontend = ?"
            params.append(frontend)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += f" ORDER BY {order_by} DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_best(
        self,
        metric: str = "best_eer",
        frontend: Optional[str] = None,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get top N experiments by a metric.

        Args:
            metric: Metric to sort by (best_eer, best_val_loss)
            frontend: Filter by frontend
            n: Number of results

        Returns:
            List of experiment dicts
        """
        query = f"""
            SELECT * FROM experiments
            WHERE status = 'completed' AND {metric} IS NOT NULL
        """
        params = []

        if frontend:
            query += " AND frontend = ?"
            params.append(frontend)

        query += f" ORDER BY {metric} ASC LIMIT ?"
        params.append(n)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def export_csv(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        **filter_kwargs,
    ) -> str:
        """
        Export experiments to CSV file.

        Args:
            path: Output CSV path
            columns: Columns to include (default: common ones)
            **filter_kwargs: Passed to get_all()

        Returns:
            Path to saved CSV
        """
        import csv

        if columns is None:
            columns = [
                "experiment_name", "frontend", "frontend_freeze",
                "lr", "weight_decay", "loss_weight_spoof", "rawboost_algo",
                "best_eer", "best_val_loss", "epochs_completed",
                "duration_seconds", "start_time", "git_hash",
            ]

        experiments = self.get_all(**filter_kwargs)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for exp in experiments:
                # Format EER as percentage
                if exp.get("best_eer") is not None:
                    exp["best_eer"] = f"{exp['best_eer']*100:.2f}%"
                # Format duration as hours
                if exp.get("duration_seconds"):
                    exp["duration_seconds"] = f"{exp['duration_seconds']/3600:.2f}h"
                writer.writerow(exp)

        return path

    def export_latex(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        caption: str = "Experiment Results",
        label: str = "tab:results",
        **filter_kwargs,
    ) -> str:
        """
        Export experiments to LaTeX table.

        Args:
            path: Output .tex path
            columns: Columns to include
            caption: Table caption
            label: LaTeX label
            **filter_kwargs: Passed to get_all()

        Returns:
            Path to saved file
        """
        if columns is None:
            columns = [
                ("frontend", "Frontend"),
                ("frontend_freeze", "Frozen"),
                ("lr", "LR"),
                ("best_eer", "EER (\\%)"),
                ("best_val_loss", "Val Loss"),
                ("epochs_completed", "Epochs"),
            ]

        experiments = self.get_all(**filter_kwargs)

        # Build LaTeX
        col_keys = [c[0] for c in columns]
        col_headers = [c[1] for c in columns]

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{" + "l" * len(columns) + "}",
            "\\toprule",
            " & ".join(col_headers) + " \\\\",
            "\\midrule",
        ]

        for exp in experiments:
            values = []
            for key in col_keys:
                val = exp.get(key, "")
                if key == "best_eer" and val is not None:
                    val = f"{val*100:.2f}"
                elif key == "best_val_loss" and val is not None:
                    val = f"{val:.4f}"
                elif key == "frontend_freeze":
                    val = "Yes" if val else "No"
                elif key == "lr" and val is not None:
                    val = f"{val:.0e}"
                else:
                    val = str(val) if val is not None else "-"
                values.append(val)
            lines.append(" & ".join(values) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def export_json(self, path: str, **filter_kwargs) -> str:
        """Export experiments to JSON file."""
        experiments = self.get_all(**filter_kwargs)

        # Parse JSON fields
        for exp in experiments:
            for field in ["config_json", "metrics_json", "gpu_info_json"]:
                if exp.get(field):
                    try:
                        exp[field.replace("_json", "")] = json.loads(exp[field])
                    except json.JSONDecodeError:
                        pass
                    del exp[field]

        with open(path, "w") as f:
            json.dump(experiments, f, indent=2, default=str)

        return path

    def summary(self) -> str:
        """Return a summary of the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Total experiments
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

            # By status
            status_counts = conn.execute("""
                SELECT status, COUNT(*) FROM experiments GROUP BY status
            """).fetchall()

            # By frontend
            frontend_counts = conn.execute("""
                SELECT frontend, COUNT(*) FROM experiments GROUP BY frontend
            """).fetchall()

            # Best EER
            best = conn.execute("""
                SELECT experiment_name, best_eer FROM experiments
                WHERE best_eer IS NOT NULL
                ORDER BY best_eer ASC LIMIT 1
            """).fetchone()

        lines = [
            f"Results Database: {self.db_path}",
            f"Total experiments: {total}",
            "",
            "By status:",
        ]
        for status, count in status_counts:
            lines.append(f"  {status}: {count}")

        lines.append("")
        lines.append("By frontend:")
        for frontend, count in frontend_counts:
            lines.append(f"  {frontend}: {count}")

        if best:
            lines.append("")
            lines.append(f"Best EER: {best[1]*100:.2f}% ({best[0]})")

        return "\n".join(lines)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
