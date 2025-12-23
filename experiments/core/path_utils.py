"""
Path utilities for unified experiment layout.

New layout (per run):
  experiments/runs/<method>/<run_id>/
    configs/    # runtime配置快照
    logs/       # stdout/stderr、LLM/token日志
    scenarios/  # 生成的场景/queue
    metrics/    # metrics_summary / records
    artifacts/  # 视频、图片、检查点等
    reports/    # 聚合后产出

保持对旧布局的兼容：优先使用新目录，若不存在则回退到旧的 experiment_results 结构。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

DEFAULT_RUN_ROOT = Path("experiments") / "runs"
LEGACY_ROOT = Path("experiment_results")


def method_dir(root: Path, method: str, run_id: str) -> Path:
    return Path(root) / method / run_id


def ensure_run_layout(run_dir: Path) -> Dict[str, Path]:
    """
    确保新布局的子目录存在，同时为旧命名创建兼容链接/目录。
    返回各子目录路径。
    """
    subdirs = {
        "configs": run_dir / "configs",
        "logs": run_dir / "logs",
        "scenarios": run_dir / "scenarios",
        "metrics": run_dir / "metrics",
        "artifacts": run_dir / "artifacts",
        "reports": run_dir / "reports",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)

    # 兼容 queue -> scenarios
    queue_dir = run_dir / "queue"
    if queue_dir.exists() and not any(subdirs["scenarios"].iterdir()):
        # 若已有 queue，保持 queue 为真实数据目录，scenarios 作为软链接
        _ensure_symlink(subdirs["scenarios"], queue_dir)
    elif not queue_dir.exists():
        # 若不存在 queue，创建并链接
        queue_dir.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(subdirs["scenarios"], queue_dir)

    # 兼容 metrics_summary 等旧文件：metrics_summary.json 位于根时，后续整理时迁移
    return {**subdirs, "queue": queue_dir}


def _ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        link_path.symlink_to(target, target_is_directory=True)
    except OSError:
        # 在不支持软链的场景下，退化为目录存在
        link_path.mkdir(parents=True, exist_ok=True)


def snapshot_args(args: Dict[str, object], dest_dir: Path) -> Path:
    """
    保存运行时参数到 configs/runtime_args.json。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    snap_path = dest_dir / "runtime_args.json"
    with snap_path.open("w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    return snap_path


def snapshot_config_file(config_path: Optional[Path], dest_dir: Path) -> Optional[Path]:
    """
    复制配置文件到 configs 目录，便于复现。
    """
    if config_path is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / config_path.name
    if config_path.exists():
        dest.write_bytes(config_path.read_bytes())
        return dest
    return None


def generate_run_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def find_run_dir(preferred_root: Path, method: str, run_id: str) -> Optional[Path]:
    """
    在新旧根目录下查找 run 目录。
    """
    new_dir = method_dir(preferred_root, method, run_id)
    if new_dir.exists():
        return new_dir
    legacy_dir = method_dir(LEGACY_ROOT, method, run_id)
    if legacy_dir.exists():
        return legacy_dir
    return None


def reorganize_legacy_files(run_dir: Path) -> None:
    """
    将旧的 metrics_summary.json / metrics_records.jsonl 等整理到 metrics 目录，
    并在根留下软链以保持兼容。
    """
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for name in ["metrics_summary.json", "metrics_records.jsonl", "token_usage.json"]:
        src = run_dir / name
        if src.exists():
            dest = metrics_dir / name
            if not dest.exists():
                src.rename(dest)
            _ensure_symlink(src, dest)


