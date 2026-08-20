from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch import Tensor
import torch

from learning.experience.transition import Transition


def _json_default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _loads(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


@dataclass
class StoredTransition:
    row_id: int
    task_id: str
    episode_id: str
    step: int
    reward: float
    success: bool
    priority: float
    metadata: dict[str, Any]


class PersistentExperienceStore:
    """SQLite-backed experience store for replay across process restarts."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    observation_json TEXT NOT NULL,
                    latent_state_json TEXT,
                    action_json TEXT NOT NULL,
                    reward REAL NOT NULL,
                    reward_components_json TEXT NOT NULL,
                    next_observation_json TEXT NOT NULL,
                    next_latent_state_json TEXT,
                    success INTEGER NOT NULL,
                    done INTEGER NOT NULL,
                    td_error REAL DEFAULT 0.0,
                    prediction_error REAL DEFAULT 0.0,
                    priority REAL DEFAULT 0.0,
                    model_versions_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_transitions_priority ON transitions(priority)")
            connection.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    def add_transition(
        self,
        *,
        task_id: str,
        episode_id: str,
        step: int,
        transition: Transition,
        action_payload: dict[str, Any],
        reward_components: dict[str, float],
        priority: float,
        model_versions: dict[str, str],
    ) -> int:
        metadata = dict(transition.metadata)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO transitions (
                    task_id, episode_id, step, observation_json, latent_state_json,
                    action_json, reward, reward_components_json, next_observation_json,
                    next_latent_state_json, success, done, td_error, prediction_error,
                    priority, model_versions_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    episode_id,
                    step,
                    _dumps(transition.observation),
                    _dumps(transition.latent_state),
                    _dumps(action_payload),
                    transition.reward,
                    _dumps(reward_components),
                    _dumps(transition.next_observation),
                    _dumps(transition.next_latent_state),
                    int(transition.success),
                    int(transition.done),
                    float(metadata.get("td_error", 0.0)),
                    float(metadata.get("prediction_error", 0.0)),
                    float(priority),
                    _dumps(model_versions),
                    _dumps(metadata),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def update_errors(self, row_id: int, td_error: float, prediction_error: float, priority: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE transitions
                SET td_error = ?, prediction_error = ?, priority = ?
                WHERE id = ?
                """,
                (float(td_error), float(prediction_error), float(priority), int(row_id)),
            )

    def recent(self, limit: int = 100) -> list[StoredTransition]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, episode_id, step, reward, success, priority, metadata_json
                FROM transitions
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            StoredTransition(
                row_id=int(row[0]),
                task_id=str(row[1]),
                episode_id=str(row[2]),
                step=int(row[3]),
                reward=float(row[4]),
                success=bool(row[5]),
                priority=float(row[6]),
                metadata=json.loads(row[7]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM transitions").fetchone()
        return int(row[0])

    def warm_start_transitions(self, limit: int = 200) -> list[Transition]:
        if limit <= 0:
            return []
        quotas = {
            "priority": max(1, int(limit * 0.40)),
            "recent": max(1, int(limit * 0.30)),
            "failures": max(1, int(limit * 0.15)),
            "successes": max(1, limit - int(limit * 0.40) - int(limit * 0.30) - int(limit * 0.15)),
        }
        queries = [
            ("priority", "ORDER BY priority DESC, id DESC", ""),
            ("recent", "ORDER BY id DESC", ""),
            ("failures", "AND success = 0 ORDER BY id DESC", ""),
            ("successes", "AND success = 1 ORDER BY id DESC", ""),
        ]
        rows_by_id: dict[int, sqlite3.Row] = {}
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            for name, ordering, predicate in queries:
                rows = connection.execute(
                    f"""
                    SELECT *
                    FROM transitions
                    WHERE 1 = 1 {predicate}
                    {ordering}
                    LIMIT ?
                    """,
                    (quotas[name],),
                ).fetchall()
                for row in rows:
                    rows_by_id[int(row["id"])] = row
        ordered_rows = sorted(rows_by_id.values(), key=lambda row: int(row["id"]))[:limit]
        return [self._row_to_transition(row) for row in ordered_rows]

    def _row_to_transition(self, row: sqlite3.Row) -> Transition:
        metadata = _loads(row["metadata_json"]) or {}
        metadata["persistent_row_id"] = int(row["id"])
        latent = self._tensor_from_json(row["latent_state_json"])
        next_latent = self._tensor_from_json(row["next_latent_state_json"])
        action_payload = _loads(row["action_json"]) or {}
        action_name = action_payload.get("action_type", 0)
        action_index = int(action_name) if isinstance(action_name, int) else metadata.get("action_type_index")
        if action_index is None:
            try:
                from environments.coding.actions import ActionType

                action_index = int(ActionType[str(action_name)])
            except Exception:
                action_index = 0
        return Transition(
            observation=_loads(row["observation_json"]),
            latent_state=latent,
            action=int(action_index),
            reward=float(row["reward"]),
            next_observation=_loads(row["next_observation_json"]),
            next_latent_state=next_latent,
            done=bool(row["done"]),
            uncertainty=float(metadata.get("uncertainty", 0.0)),
            novelty=float(metadata.get("novelty", 0.0)),
            success=bool(row["success"]),
            metadata=metadata,
        )

    @staticmethod
    def _tensor_from_json(value: str | None) -> Tensor | None:
        loaded = _loads(value)
        if loaded is None:
            return None
        return torch.tensor(loaded, dtype=torch.float32)

    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else 0
