#!/usr/bin/env python3
"""
Incremental local diversity comparison (LRD/SCD/TER; legacy keys LMS/SED/OSCR)
for GPT-guided vs Random.

Given two experiment directories (queue/ with json or pkl scenarios), this script:
1) Loads scenarios from each directory.
2) Builds incremental checkpoints (by scenario count).
3) Computes LMS/SED/OSCR (interpreted as LRD/SCD/TER) for each prefix.
4) Saves an incremental JSON and a PNG plot with three curves (LRD/SCD/TER, keys unchanged).

Example:
python -m experiments.analysis.incremental_local_diversity \
  --gpt-dir ./experiment_results/ScenarioFuzz-LLM/scenariofuzz-llm_20251224_170025 \
  --rand-dir ./experiment_results/ScenarioFuzz-LLM/scenariofuzz-llm_20251225_073610 \
  --output-dir ./experiment_results \
  --num-checkpoints 10
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

from experiments.analysis.local_diversity_metrics import summarize_local_diversity
from experiments.analysis.scenario_loader import ScenarioDataLoader

TITLE_SIZE = 10.5  # 五号
LABEL_SIZE = 9     # 小五


HOME = Path.home()

CHINESE_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Regular.otf")),
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Medium.otf")),
    ("Noto Serif CJK SC", str(HOME / ".local/share/fonts/custom/NotoSerifSC-Bold.otf")),
    ("Noto Serif CJK SC", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ("DejaVu Serif", ""),
]

LATIN_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("DejaVu Serif", ""),
]


def _select_font(candidates: List[Tuple[str, str]]) -> str:
    for name, path in candidates:
        if path:
            p = Path(path)
            if p.exists():
                try:
                    font_manager.fontManager.addfont(str(p))
                    real_name = font_manager.FontProperties(fname=str(p)).get_name()
                    if real_name:
                        return real_name
                except Exception:
                    pass
        try:
            found_path = font_manager.findfont(name, fallback_to_default=False)
            if found_path:
                return name
        except Exception:
            continue
    print(f"[Font] 未找到候选字体，使用 {candidates[-1][0]} 作为兜底")
    return candidates[-1][0]


def configure_fonts():
    chinese = _select_font(CHINESE_FONT_CANDIDATES)
    latin = _select_font(LATIN_FONT_CANDIDATES)
    rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": [chinese, latin, "serif"],
            "font.serif": [latin, chinese, "DejaVu Serif"],
            "font.sans-serif": [chinese, latin, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": LABEL_SIZE,
            "ytick.labelsize": LABEL_SIZE,
            "legend.fontsize": LABEL_SIZE,
        }
    )


def _sort_scenarios(scenarios: List) -> List:
    """Stable sorting: prefer scenario_id if exists, else filename order is preserved by loader."""
    def _key(sc):
        sid = getattr(sc, "scenario_id", None)
        return sid if sid is not None else math.inf
    with_id = [s for s in scenarios if getattr(s, "scenario_id", None) is not None]
    without_id = [s for s in scenarios if getattr(s, "scenario_id", None) is None]
    return sorted(with_id, key=_key) + without_id


def _build_checkpoints(max_len: int, num_checkpoints: int, step: int, custom: Sequence[int]) -> List[int]:
    if max_len <= 0:
        return []
    if custom:
        cps = sorted({c for c in custom if 1 <= c <= max_len})
        return cps
    if step and step > 0:
        cps = list(range(step, max_len + 1, step))
        if cps[-1] != max_len:
            cps.append(max_len)
        return cps
    # even spacing
    num = max(1, num_checkpoints)
    cps = []
    for i in range(1, num + 1):
        cps.append(max(1, round(i * max_len / num)))
    # dedup and sort
    return sorted(set(cps))


def _compute_series(scenarios: List, checkpoints: List[int]) -> List[Dict]:
    series = []
    for n in checkpoints:
        prefix = scenarios[:n]
        metrics = summarize_local_diversity(prefix)
        metrics["count"] = n
        series.append(metrics)
    return series


def _plot_series(
    checkpoints: List[int],
    gpt_series: List[Dict],
    rand_series: List[Dict],
    output_pdf: Path,
):
    configure_fonts()
    x = checkpoints
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("局部多样性增量对比（LRD / SCD / TER）", fontweight="bold")

    def _plot(ax, key, title):
        gpt_y = [d.get(key, 0.0) for d in gpt_series]
        rand_y = [d.get(key, 0.0) for d in rand_series]
        ax.plot(x, gpt_y, label="GPT 引导", linewidth=1.2)
        ax.plot(x, rand_y, label="随机变异", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("场景前缀数")
        ax.grid(True, linestyle="--", alpha=0.5)

    _plot(axes[0], "lms", "LRD：种子内部扩散")
    _plot(axes[1], "sed", "SCD：相对种子偏移")
    _plot(axes[2], "oscr", "TER：典型集逃逸率")
    axes[2].legend()

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _shorten_dir(path: Path) -> str:
    """Return anonymized dir label (last two components) to avoid leaking absolute paths."""
    parts = path.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return path.name


def main():
    parser = argparse.ArgumentParser(description="Incremental local diversity comparison.")
    parser.add_argument("--gpt-dir", required=True, help="GPT-guided experiment dir (contains queue/)")
    parser.add_argument("--rand-dir", required=True, help="Random mutation experiment dir (contains queue/)")
    parser.add_argument("--output-dir", default="./experiment_results", help="Directory to store incremental outputs")
    parser.add_argument("--num-checkpoints", type=int, default=10, help="Number of evenly spaced checkpoints")
    parser.add_argument("--step", type=int, default=0, help="Fixed step size for checkpoints (overrides num-checkpoints if >0)")
    parser.add_argument("--checkpoints", type=str, default="", help="Custom checkpoints comma-separated, e.g., 10,20,50")
    parser.add_argument("--prefer-pickle", action="store_true", help="Prefer pickle when loading scenarios")
    args = parser.parse_args()

    gpt_dir = Path(args.gpt_dir).resolve()
    rand_dir = Path(args.rand_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gpt_dir.exists():
        raise FileNotFoundError(f"GPT dir not found: {gpt_dir}")
    if not rand_dir.exists():
        raise FileNotFoundError(f"Random dir not found: {rand_dir}")

    loader = ScenarioDataLoader()
    gpt_scenarios = loader.load_all_scenarios(gpt_dir, prefer_pickle=args.prefer_pickle)
    rand_scenarios = loader.load_all_scenarios(rand_dir, prefer_pickle=args.prefer_pickle)

    gpt_scenarios = _sort_scenarios(gpt_scenarios)
    rand_scenarios = _sort_scenarios(rand_scenarios)

    max_len = min(len(gpt_scenarios), len(rand_scenarios))
    if max_len == 0:
        raise RuntimeError("No scenarios to compare (one of the dirs is empty)")

    custom_cps = [int(x) for x in args.checkpoints.split(",") if x.strip().isdigit()] if args.checkpoints else []
    checkpoints = _build_checkpoints(max_len=max_len, num_checkpoints=args.num_checkpoints, step=args.step, custom=custom_cps)

    gpt_series = _compute_series(gpt_scenarios, checkpoints)
    rand_series = _compute_series(rand_scenarios, checkpoints)

    result = {
        "gpt_dir": _shorten_dir(gpt_dir),
        "rand_dir": _shorten_dir(rand_dir),
        "checkpoints": checkpoints,
        "gpt": gpt_series,
        "random": rand_series,
    }

    output_json = out_dir / "local_diversity_incremental.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    output_pdf = out_dir / "local_diversity_incremental.pdf"
    _plot_series(checkpoints, gpt_series, rand_series, output_pdf)

    print(f"[Incremental] Saved JSON -> {output_json}")
    print(f"[Incremental] Saved plot -> {output_pdf}")
    print(f"[Incremental] Checkpoints: {checkpoints}")


if __name__ == "__main__":
    main()

