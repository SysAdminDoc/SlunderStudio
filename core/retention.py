"""
Slunder Studio - Recovery artifact retention.
Jobs, crash logs, settings backups and project versions all accumulate. This
module applies age, count and size policies to each of them, always with a
dry-run preview first, and never prunes a record that is still active or that
a user could still recover from.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from core.settings import Settings, get_config_dir

# Categories the recovery center manages.
CATEGORY_JOBS = "jobs"
CATEGORY_JOB_LOGS = "job_logs"
CATEGORY_CRASH_LOGS = "crash_logs"
CATEGORY_SETTINGS_BACKUPS = "settings_backups"
CATEGORY_PROJECT_VERSIONS = "project_versions"

CATEGORIES = (
    CATEGORY_JOBS,
    CATEGORY_JOB_LOGS,
    CATEGORY_CRASH_LOGS,
    CATEGORY_SETTINGS_BACKUPS,
    CATEGORY_PROJECT_VERSIONS,
)

CATEGORY_LABELS = {
    CATEGORY_JOBS: "Completed tasks",
    CATEGORY_JOB_LOGS: "Task logs",
    CATEGORY_CRASH_LOGS: "Crash logs",
    CATEGORY_SETTINGS_BACKUPS: "Settings backups",
    CATEGORY_PROJECT_VERSIONS: "Project versions",
}


@dataclass(frozen=True)
class RetentionPolicy:
    """Age, count and size limits for one category. Zero means no limit."""
    category: str
    max_age_days: float = 0.0
    max_count: int = 0
    max_total_mb: float = 0.0

    def describe(self) -> str:
        parts = []
        if self.max_age_days:
            parts.append(f"older than {self.max_age_days:g} days")
        if self.max_count:
            parts.append(f"beyond the newest {self.max_count}")
        if self.max_total_mb:
            parts.append(f"over {self.max_total_mb:g} MB total")
        return ", ".join(parts) or "no limit"


# Safe defaults: generous enough that normal use never loses anything, tight
# enough that these directories cannot grow without bound.
DEFAULT_POLICIES = {
    CATEGORY_JOBS: RetentionPolicy(CATEGORY_JOBS, max_age_days=30, max_count=500),
    CATEGORY_JOB_LOGS: RetentionPolicy(
        CATEGORY_JOB_LOGS, max_age_days=30, max_count=500, max_total_mb=200
    ),
    CATEGORY_CRASH_LOGS: RetentionPolicy(
        CATEGORY_CRASH_LOGS, max_age_days=90, max_count=50, max_total_mb=50
    ),
    CATEGORY_SETTINGS_BACKUPS: RetentionPolicy(
        CATEGORY_SETTINGS_BACKUPS, max_age_days=90, max_count=40, max_total_mb=20
    ),
    CATEGORY_PROJECT_VERSIONS: RetentionPolicy(
        CATEGORY_PROJECT_VERSIONS, max_count=20
    ),
}

SETTINGS_PREFIX = "retention"


@dataclass(frozen=True)
class RetentionItem:
    """One prunable artifact."""
    category: str
    identifier: str
    label: str
    timestamp: float
    size_bytes: int = 0
    path: str = ""
    protected: bool = False
    protected_reason: str = ""


@dataclass
class RetentionPlan:
    """What a cleanup would remove, before anything is removed."""
    category: str
    policy: RetentionPolicy
    keep: list[RetentionItem] = field(default_factory=list)
    remove: list[RetentionItem] = field(default_factory=list)
    protected: list[RetentionItem] = field(default_factory=list)

    @property
    def removed_bytes(self) -> int:
        return sum(item.size_bytes for item in self.remove)

    def summary(self) -> str:
        if not self.remove:
            return (
                f"{CATEGORY_LABELS.get(self.category, self.category)}: nothing to remove "
                f"({len(self.keep)} kept, {len(self.protected)} protected)"
            )
        return (
            f"{CATEGORY_LABELS.get(self.category, self.category)}: "
            f"remove {len(self.remove)} ({self.removed_bytes / 1e6:.1f} MB), "
            f"keep {len(self.keep)}, protect {len(self.protected)}"
        )


def load_policy(category: str, settings: Optional[Settings] = None) -> RetentionPolicy:
    """Read a category's policy from settings, falling back to the default."""
    default = DEFAULT_POLICIES.get(category, RetentionPolicy(category))
    settings = settings or Settings()

    def _number(field_name: str, fallback: float) -> float:
        raw = settings.get(f"{SETTINGS_PREFIX}.{category}.{field_name}", None)
        if raw is None:
            return fallback
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return fallback
        return max(0.0, value)

    return RetentionPolicy(
        category=category,
        max_age_days=_number("max_age_days", default.max_age_days),
        max_count=int(_number("max_count", default.max_count)),
        max_total_mb=_number("max_total_mb", default.max_total_mb),
    )


def plan_cleanup(items: Iterable[RetentionItem], policy: RetentionPolicy,
                 now: Optional[float] = None) -> RetentionPlan:
    """Decide what would be removed. Never selects a protected item."""
    now = time.time() if now is None else now
    plan = RetentionPlan(category=policy.category, policy=policy)

    candidates = []
    for item in items:
        if item.protected:
            plan.protected.append(item)
        else:
            candidates.append(item)

    # Newest first: age and count limits both keep the most recent.
    candidates.sort(key=lambda item: item.timestamp, reverse=True)

    survivors: list[RetentionItem] = []
    for index, item in enumerate(candidates):
        too_old = (
            policy.max_age_days > 0
            and (now - item.timestamp) > policy.max_age_days * 86400
        )
        too_many = policy.max_count > 0 and index >= policy.max_count
        if too_old or too_many:
            plan.remove.append(item)
        else:
            survivors.append(item)

    if policy.max_total_mb > 0:
        budget = policy.max_total_mb * 1_000_000
        running = 0
        kept: list[RetentionItem] = []
        for item in survivors:
            running += item.size_bytes
            if running > budget and kept:
                plan.remove.append(item)
            else:
                kept.append(item)
        survivors = kept

    plan.keep = survivors
    return plan


# ── Collectors ─────────────────────────────────────────────────────────────────

def collect_job_items(job_store) -> list[RetentionItem]:
    """Finished job records. Active and recoverable jobs are protected."""
    from core.job_state import ACTIVE_STATUSES, JobStatus

    protected_statuses = set(ACTIVE_STATUSES) | {JobStatus.RECOVERABLE}
    items = []
    for record in job_store.list_records():
        protected = record.status in protected_statuses
        items.append(RetentionItem(
            category=CATEGORY_JOBS,
            identifier=record.id,
            label=f"{record.kind}: {record.label}",
            timestamp=float(record.updated_at or record.created_at or 0.0),
            protected=protected,
            protected_reason=(
                f"status is {record.status}" if protected else ""
            ),
        ))
    return items


def _file_items(category: str, paths: Iterable[Path],
                protected_paths: Iterable[Path] = ()) -> list[RetentionItem]:
    protected = {Path(p).resolve() for p in protected_paths}
    items = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        is_protected = path.resolve() in protected
        items.append(RetentionItem(
            category=category,
            identifier=path.name,
            label=path.name,
            timestamp=stat.st_mtime,
            size_bytes=stat.st_size,
            path=str(path),
            protected=is_protected,
            protected_reason="current file" if is_protected else "",
        ))
    return items


def collect_job_log_items(root: Optional[Path] = None) -> list[RetentionItem]:
    directory = Path(root) if root else get_config_dir() / "jobs" / "logs"
    if not directory.is_dir():
        return []
    return _file_items(CATEGORY_JOB_LOGS, sorted(directory.glob("*.log")))


def collect_crash_log_items(root: Optional[Path] = None) -> list[RetentionItem]:
    directory = Path(root) if root else get_config_dir()
    if not directory.is_dir():
        return []
    current = directory / "crash.log"
    candidates = sorted(directory.glob("crash*.log"))
    return _file_items(CATEGORY_CRASH_LOGS, candidates, protected_paths=[current])


def collect_settings_backup_items(root: Optional[Path] = None) -> list[RetentionItem]:
    directory = Path(root) if root else get_config_dir() / "backups"
    if not directory.is_dir():
        return []
    return _file_items(CATEGORY_SETTINGS_BACKUPS, sorted(directory.glob("*.bak")))


def collect_project_version_items(project_manager) -> list[RetentionItem]:
    """Versions of the open project. Pre-restore snapshots are protected."""
    from core.project import PROTECTED_VERSION_KINDS

    project = project_manager.current
    if project is None:
        return []
    items = []
    newest = max((v.version for v in project.versions), default=0)
    for version in project.versions:
        path = project_manager.version_dir(project.id, version.version)
        size = 0
        for file in path.rglob("*") if path.is_dir() else []:
            if file.is_file():
                size += file.stat().st_size
        protected = version.kind in PROTECTED_VERSION_KINDS or version.version == newest
        items.append(RetentionItem(
            category=CATEGORY_PROJECT_VERSIONS,
            identifier=str(version.version),
            label=f"v{version.version} ({version.label}) {version.description}",
            timestamp=version.timestamp,
            size_bytes=size,
            path=str(path),
            protected=protected,
            protected_reason=(
                "pre-restore snapshot" if version.kind in PROTECTED_VERSION_KINDS
                else "newest version" if version.version == newest else ""
            ),
        ))
    return items


# ── Execution ──────────────────────────────────────────────────────────────────

def delete_path_item(item: RetentionItem) -> bool:
    path = Path(item.path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            return False
    except OSError:
        return False
    return True


def apply_plan(plan: RetentionPlan,
               remover: Optional[Callable[[RetentionItem], bool]] = None) -> list[RetentionItem]:
    """Execute a plan. Returns the items actually removed."""
    remover = remover or delete_path_item
    removed = []
    for item in plan.remove:
        if remover(item):
            removed.append(item)
    return removed


class RecoveryCenter:
    """One place to inspect, preview, and prune every recovery artifact."""

    def __init__(self, settings: Optional[Settings] = None,
                 job_store=None, project_manager=None):
        self._settings = settings or Settings()
        self._job_store = job_store
        self._project_manager = project_manager

    @property
    def job_store(self):
        if self._job_store is None:
            from core.job_state import JobStore

            self._job_store = JobStore()
        return self._job_store

    @property
    def project_manager(self):
        if self._project_manager is None:
            from core.project import get_project_manager

            self._project_manager = get_project_manager()
        return self._project_manager

    def collect(self, category: str) -> list[RetentionItem]:
        if category == CATEGORY_JOBS:
            return collect_job_items(self.job_store)
        if category == CATEGORY_JOB_LOGS:
            return collect_job_log_items()
        if category == CATEGORY_CRASH_LOGS:
            return collect_crash_log_items()
        if category == CATEGORY_SETTINGS_BACKUPS:
            return collect_settings_backup_items()
        if category == CATEGORY_PROJECT_VERSIONS:
            return collect_project_version_items(self.project_manager)
        raise ValueError(f"Unknown retention category: {category}")

    def preview(self, category: str, now: Optional[float] = None) -> RetentionPlan:
        """Dry run. Nothing is removed."""
        policy = load_policy(category, self._settings)
        return plan_cleanup(self.collect(category), policy, now=now)

    def preview_all(self, now: Optional[float] = None) -> dict[str, RetentionPlan]:
        return {category: self.preview(category, now=now) for category in CATEGORIES}

    def clean(self, category: str, now: Optional[float] = None) -> list[RetentionItem]:
        plan = self.preview(category, now=now)
        if category == CATEGORY_JOBS:
            return apply_plan(plan, self._remove_job_record)
        if category == CATEGORY_PROJECT_VERSIONS:
            return apply_plan(plan, self._remove_project_version)
        return apply_plan(plan)

    def clean_all(self, now: Optional[float] = None) -> dict[str, list[RetentionItem]]:
        return {category: self.clean(category, now=now) for category in CATEGORIES}

    def _remove_job_record(self, item: RetentionItem) -> bool:
        return bool(self.job_store.delete(item.identifier))

    def _remove_project_version(self, item: RetentionItem) -> bool:
        manager = self.project_manager
        project = manager.current
        if project is None:
            return False
        version = int(item.identifier)
        path = manager.version_dir(project.id, version)
        try:
            if path.is_dir():
                shutil.rmtree(path)
        except OSError:
            return False
        project.versions = [v for v in project.versions if v.version != version]
        manager.save()
        return True

    def total_size_bytes(self) -> int:
        return sum(
            item.size_bytes
            for category in CATEGORIES
            for item in self.collect(category)
        )
