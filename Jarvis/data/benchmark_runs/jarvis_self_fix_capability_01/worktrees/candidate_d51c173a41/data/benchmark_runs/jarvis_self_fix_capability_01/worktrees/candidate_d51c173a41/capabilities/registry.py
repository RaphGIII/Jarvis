from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capabilities.models import CapabilityManifest


class CapabilityRegistry:
    """Persistent registry for installed Jarvis capabilities."""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, CapabilityManifest] = {}
        self._load()

    def has(self, capability_id: str) -> bool:
        manifest = self._records.get(capability_id)
        return bool(manifest and manifest.status == "active")

    def get(self, capability_id: str) -> CapabilityManifest | None:
        return self._records.get(capability_id)

    def find(self, query: str, *, limit: int = 5) -> list[CapabilityManifest]:
        query_terms = self._terms(query)
        scored: list[tuple[float, CapabilityManifest]] = []
        for manifest in self._records.values():
            if manifest.status != "active":
                continue
            haystack = self._terms(f"{manifest.capability_id} {manifest.description}")
            overlap = len(query_terms & haystack)
            if manifest.capability_id.lower() in query.lower():
                overlap += 4
            if overlap:
                scored.append((overlap / max(1, len(query_terms)), manifest))
        scored.sort(key=lambda item: (item[0], item[1].capability_id), reverse=True)
        return [manifest for _, manifest in scored[:limit]]

    def register(self, manifest: CapabilityManifest) -> CapabilityManifest:
        errors = manifest.validate()
        if errors:
            raise ValueError("; ".join(errors))
        existing = self._records.get(manifest.capability_id)
        if existing and existing.status == "active" and existing.version == manifest.version:
            raise ValueError(f"Capability already registered at version {manifest.version}: {manifest.capability_id}")
        self._records[manifest.capability_id] = manifest
        self._save()
        return manifest

    def disable(self, capability_id: str, reason: str = "") -> CapabilityManifest:
        manifest = self._records[capability_id]
        manifest.status = "disabled"
        manifest.validation_status = {**manifest.validation_status, "disabled_reason": reason}
        self._save()
        return manifest

    def all(self) -> list[CapabilityManifest]:
        return list(self._records.values())

    def _load(self) -> None:
        if not self.path.exists():
            self._records = {}
            self._save()
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) > self.schema_version:
            raise ValueError("Unsupported capability registry schema version.")
        self._records = {
            capability_id: CapabilityManifest.from_dict(payload)
            for capability_id, payload in (data.get("capabilities") or {}).items()
        }

    def _save(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "capabilities": {capability_id: manifest.to_dict() for capability_id, manifest in sorted(self._records.items())},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term for term in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(term) > 2}
