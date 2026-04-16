#!/usr/bin/env python3
"""
根据 evaluation_results 中的评测得分，过滤 metadatav1.csv 中评分不好的条目，
生成新的 metadatav2.csv。

评测维度（5个）:
  - motion_smoothness:  运动平滑度，范围 ~0.93-1.0，越高越好
  - dynamic_degree:     动态程度，True/False (1/0)
  - aesthetic_quality:   美学质量，范围 ~0.20-0.65，越高越好
  - imaging_quality:     成像质量，范围 ~32-81（原始分数），越高越好
  - temporal_flickering: 时间闪烁，范围 ~0.90-1.0，越高越好（越少闪烁）

过滤规则:
  1. dynamic_degree 必须为 True（视频需要有足够的运动）
  2. motion_smoothness >= 0.96（运动需要平滑）
  3. aesthetic_quality >= 0.35（美学质量不能太差）
  4. imaging_quality >= 60（成像质量不能太低）
  5. temporal_flickering >= 0.94（闪烁不能太严重）

用法:
  python filter_by_eval_scores.py [--dry-run] [--output OUTPUT_CSV]

  --dry-run           只打印统计信息，不写出文件
  --output OUTPUT     输出 CSV 路径（默认: metadatav2.csv，与输入同目录）
  --save-report FILE  保存过滤报告到指定 JSON 文件
"""

import argparse
import csv
import json
import glob
import os
from collections import defaultdict


# ======================== 过滤阈值配置 ========================
THRESHOLDS = {
    "motion_smoothness": 0.96,    # 运动平滑度下限
    "dynamic_degree": True,       # 必须有动态
    "aesthetic_quality": 0.35,    # 美学质量下限
    "imaging_quality": 60.0,     # 成像质量下限
    "temporal_flickering": 0.94,  # 时间稳定性下限
}

# 类别名称映射: eval_results 目录名 -> VACEv2 CSV 中的目录名
CATEGORY_MAP = {
    "shoe": "shoes",
    "human": "human",
    "makeup": "makeup",
    "clothes_accessory": "clothes_accessory",
    "jewelry": "jewelry",
}

EVAL_BASE = "/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/evaluation_results"
DEFAULT_METADATA_CSV = "/mmu_mllm_hdd_2/jinlv/VideoEditing/data/Custom/VACEv2/metadatav1.csv"


def load_all_eval_results():
    """加载所有类别的评测结果，返回 {vace_category: {video_name: {metric: score}}}"""
    all_scores = {}

    for eval_cat, vace_cat in CATEGORY_MAP.items():
        cat_base = os.path.join(EVAL_BASE, eval_cat)
        eval_files = sorted(glob.glob(os.path.join(cat_base, "**/*_eval_results.json"), recursive=True))

        if not eval_files:
            print(f"[WARNING] 未找到 {eval_cat} 的评测结果文件")
            continue

        video_scores = defaultdict(dict)
        for eval_file in eval_files:
            with open(eval_file, "r") as f:
                data = json.load(f)

            for metric, (mean_val, video_list) in data.items():
                for item in video_list:
                    video_path = item["video_path"]
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    score = item["video_results"]
                    if metric == "dynamic_degree":
                        score = bool(score)
                    video_scores[video_name][metric] = score

        all_scores[vace_cat] = dict(video_scores)
        print(f"[INFO] {eval_cat} -> {vace_cat}: 加载了 {len(video_scores)} 个视频的评分")

    return all_scores


def check_video_pass(scores_dict):
    """检查单个视频是否通过所有过滤条件。返回 (pass, fail_reasons)"""
    fail_reasons = []

    dd = scores_dict.get("dynamic_degree")
    if THRESHOLDS["dynamic_degree"] and dd is not None and not dd:
        fail_reasons.append("dynamic_degree=False")

    for metric in ["motion_smoothness", "aesthetic_quality", "imaging_quality", "temporal_flickering"]:
        val = scores_dict.get(metric)
        if val is not None and val < THRESHOLDS[metric]:
            fail_reasons.append(f"{metric}={val:.4f}<{THRESHOLDS[metric]}")

    return len(fail_reasons) == 0, fail_reasons


def extract_video_info(video_path):
    """
    从 CSV 的 video 列提取类别和视频名。
    路径格式: ../../data/Custom/VACEv2/shoes/943708146956-Scene-001/orig.mp4
    返回 (category, video_name) 如 ("shoes", "943708146956-Scene-001")
    """
    parts = video_path.split("/")
    try:
        idx = parts.index("VACEv2")
        category = parts[idx + 1]
        video_name = parts[idx + 2]
        return category, video_name
    except (ValueError, IndexError):
        return None, None


def filter_metadata(all_scores, input_csv, dry_run=False, output_csv=None):
    """读取输入 CSV，过滤掉评分不好的行，输出新 CSV"""

    total = 0
    kept = 0
    removed = 0
    no_eval = 0

    category_stats = defaultdict(lambda: {"total": 0, "kept": 0, "removed": 0, "no_eval": 0})
    fail_metric_counts = defaultdict(lambda: defaultdict(int))

    kept_rows = []

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        kept_rows.append(header)

        for row in reader:
            total += 1
            video_path = row[0]  # video 列
            category, video_name = extract_video_info(video_path)

            if category is None:
                # 无法解析路径，保留
                kept_rows.append(row)
                kept += 1
                continue

            cat_stats = category_stats[category]
            cat_stats["total"] += 1

            scores = all_scores.get(category, {})
            if video_name not in scores:
                # 无评分数据，保留
                kept_rows.append(row)
                kept += 1
                no_eval += 1
                cat_stats["kept"] += 1
                cat_stats["no_eval"] += 1
                continue

            passed, reasons = check_video_pass(scores[video_name])
            if passed:
                kept_rows.append(row)
                kept += 1
                cat_stats["kept"] += 1
            else:
                removed += 1
                cat_stats["removed"] += 1
                for r in reasons:
                    metric_name = r.split("=")[0]
                    fail_metric_counts[category][metric_name] += 1

    # 打印统计
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(f"\n{'='*60}")
        print(f"类别: {category}")
        print(f"  CSV 条目总数:   {stats['total']}")
        print(f"  保留:           {stats['kept']}")
        print(f"  过滤掉:         {stats['removed']}")
        print(f"  无评分(保留):   {stats['no_eval']}")
        if stats["total"] > 0:
            print(f"  保留率:         {stats['kept'] / stats['total'] * 100:.1f}%")
        if fail_metric_counts[category]:
            print(f"  各维度淘汰数:")
            for metric, count in sorted(fail_metric_counts[category].items(), key=lambda x: -x[1]):
                print(f"    {metric}: {count}")

    print(f"\n{'='*60}")
    print(f"总计:")
    print(f"  CSV 条目总数:   {total}")
    print(f"  保留:           {kept}")
    print(f"  过滤掉:         {removed}")
    print(f"  无评分(保留):   {no_eval}")
    if total > 0:
        print(f"  保留率:         {kept / total * 100:.1f}%")

    print(f"\n保留的视频数量: {kept}")

    # 写出结果
    if not dry_run and output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in kept_rows:
                writer.writerow(row)
        print(f"\n已写出过滤后的 CSV: {output_csv}")
        print(f"  行数: {len(kept_rows) - 1} 条数据 + 1 行表头")

    return {
        "thresholds": {k: str(v) for k, v in THRESHOLDS.items()},
        "category_stats": {k: dict(v) for k, v in category_stats.items()},
        "total": {"total": total, "kept": kept, "removed": removed, "no_eval": no_eval},
    }


def main():
    parser = argparse.ArgumentParser(description="根据评测得分过滤 metadata CSV")
    parser.add_argument("--input", type=str, default=DEFAULT_METADATA_CSV,
                        help=f"输入 CSV 路径（默认: {DEFAULT_METADATA_CSV}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计信息，不写出文件")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 路径（默认: 与输入同目录下的 metadatav2.csv）")
    parser.add_argument("--save-report", type=str, default=None,
                        help="保存过滤报告到指定 JSON 文件")

    # 自定义阈值
    parser.add_argument("--th-motion-smoothness", type=float, default=None)
    parser.add_argument("--th-aesthetic-quality", type=float, default=None)
    parser.add_argument("--th-imaging-quality", type=float, default=None)
    parser.add_argument("--th-temporal-flickering", type=float, default=None)
    parser.add_argument("--no-filter-dynamic", action="store_true",
                        help="不过滤 dynamic_degree")

    args = parser.parse_args()

    if args.th_motion_smoothness is not None:
        THRESHOLDS["motion_smoothness"] = args.th_motion_smoothness
    if args.th_aesthetic_quality is not None:
        THRESHOLDS["aesthetic_quality"] = args.th_aesthetic_quality
    if args.th_imaging_quality is not None:
        THRESHOLDS["imaging_quality"] = args.th_imaging_quality
    if args.th_temporal_flickering is not None:
        THRESHOLDS["temporal_flickering"] = args.th_temporal_flickering
    if args.no_filter_dynamic:
        THRESHOLDS["dynamic_degree"] = False

    # 默认输出路径
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "metadatav2.csv")

    print("过滤阈值配置:")
    for k, v in THRESHOLDS.items():
        print(f"  {k}: {v}")
    print(f"\n输入: {args.input}")
    print(f"输出: {args.output}")
    print()

    # 加载评测结果
    print("正在加载评测结果...")
    all_scores = load_all_eval_results()

    # 过滤
    result = filter_metadata(all_scores, input_csv=args.input, dry_run=args.dry_run, output_csv=args.output)

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存到: {args.save_report}")


if __name__ == "__main__":
    main()
