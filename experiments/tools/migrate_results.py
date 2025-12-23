#!/usr/bin/env python3
"""
将旧的 experiment_results 目录索引到新的 experiments/runs 布局。
默认行为：为每个旧 run 创建指向新布局的目录（使用软链接或直接复用），
同时补齐新布局的子目录并整理 metrics/queue 等。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiments.core.path_utils import ensure_run_layout, reorganize_legacy_files


def migrate_one(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        # 软链失败则复制目录（不深度复制大型数据）
        shutil.copytree(src, dst)
    ensure_run_layout(dst)
    reorganize_legacy_files(dst)


def migrate_all(legacy_root: Path, new_root: Path) -> None:
    if not legacy_root.exists():
        print(f"[INFO] legacy root not found: {legacy_root}")
        return
    for method_dir in legacy_root.iterdir():
        if not method_dir.is_dir():
            continue
        for run_dir in method_dir.iterdir():
            if not run_dir.is_dir():
                continue
            target = new_root / method_dir.name / run_dir.name
            print(f"[INFO] migrating {run_dir} -> {target}")
            migrate_one(run_dir, target)


def main():
    parser = argparse.ArgumentParser(description="迁移旧 experiment_results 目录到新布局")
    parser.add_argument("--legacy-root", default="experiment_results", help="旧目录根路径")
    parser.add_argument("--new-root", default="experiments/runs", help="新目录根路径")
    args = parser.parse_args()

    migrate_all(Path(args.legacy_root), Path(args.new_root))


if __name__ == "__main__":
    main()

