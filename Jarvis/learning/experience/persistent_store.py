from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch import Tensor

from learning.experience.transition import Transition


def _json_default(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


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
