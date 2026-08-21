from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from capabilities.models import CapabilityManifest, SkillSpecification
from capabilities.registry import CapabilityRegistry
from capabilities.workspace import StagedSkillWorkspace


@dataclass
class PromotionDecision:
    promoted: bool
    manifest: CapabilityManifest | None = None
    errors: list[str] = field(default_factory=list)


class SkillPromoter:
    """Validates and atomically promotes staged skills into the installed catalog."""

    def __init__(self, installed_root: str | Path, registry: CapabilityRegistry) -> None:
        self.installed_root = Path(installed_root)
        self.registry = registry
        self.installed_root.mkdir(parents=True, exist_ok=True)

    def promote(
        self,
        spec: SkillSpecification,
        staged: StagedSkillWorkspace,
        *,
        public_success: bool,
        hidden_success: bool,
    ) -> PromotionDecision:
        errors = []
        if not public_success:
            errors.append("public tests did not pass")
        if not hidden_success:
            errors.append("hidden verifier did not pass")
        if not staged.protected_files_pristine():
            errors.append("protected public tests or skill specification were modified")
        entrypoint = staged.root / "main.py"
        if not entrypoint.exists():
            errors.append("entrypoint main.py does not exist")
        manifest = spec.to_manifest()
        manifest_errors = manifest.validate()
        errors.extend(manifest_errors)
        target = self.installed_root / manifest.capability_id / manifest.version
        if target.exists():
            errors.append(f"installed version already exists: {target}")
        if errors:
            return PromotionDecision(False, None, errors)

        tmp = target.parent / f".{manifest.version}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged.root, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        tmp.rename(target)

        manifest.source_location = str(target.resolve())
        manifest.tests_location = str((target / "test_public.py").resolve())
        manifest.validation_status = {
            "syntax_build": True,
            "public_tests": True,
            "hidden_verifier": True,
            "protected_files_pristine": True,
            "permission_policy": True,
        }
        registry_manifest = self.registry.register(manifest)
        return PromotionDecision(True, registry_manifest, [])
