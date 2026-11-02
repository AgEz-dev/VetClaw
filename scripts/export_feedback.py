"""拉取医院侧已标注反馈（JSONL），合并进本地评测集。

用法：
    python -m scripts.export_feedback
    python -m scripts.export_feedback --output datasets/eval_feedback.jsonl

约定：
- 医院侧**导出即消费**（LABELED → EXPORTED），同一批样本不会返回第二次；
  本地按 `id` 去重追加，网络抖动时重跑不会产生重复用例。
- 样本字段即评测集契约（id / query / expected_source_chunk / must_contain_keywords /
  expected_behavior），外加 source_feedback_id / labeler / labeled_at 溯源信息。
- 本地文件含真实问答文本，默认被 .gitignore 排除，不进仓库。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hospital_client import HospitalClient, HospitalClientError  # noqa: E402

DEFAULT_OUTPUT = Path("datasets/eval_feedback.jsonl")
REQUIRED_FIELDS = ("id", "query", "expected_behavior")


def parse_jsonl(text: str) -> list[dict]:
    """逐行解析并校验评测字段；行号从 1 开始，方便定位医院侧导出问题。"""
    samples = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"第 {lineno} 行不是合法 JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"第 {lineno} 行不是 JSON 对象")
        missing = [key for key in REQUIRED_FIELDS if not item.get(key)]
        if missing:
            raise ValueError(f"第 {lineno} 行缺少字段: {missing}")
        samples.append(item)
    return samples


def merge_samples(path: Path, samples: list[dict]) -> int:
    """按 id 去重追加，返回新增条数；已有文件缺字段会直接报错而不是静默跳过。"""
    seen = set()
    if path.exists():
        seen = {item.get("id") for item in
                parse_jsonl(path.read_text(encoding="utf-8"))}
    fresh = [item for item in samples if item["id"] not in seen]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in fresh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(fresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取医院侧已标注反馈（JSONL）")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"本地评测集路径（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--base-url", default=None,
                        help="医院服务地址；缺省读 VETCLAW_HOSPITAL_BASE_URL")
    args = parser.parse_args(argv)

    client = HospitalClient(base_url=args.base_url) if args.base_url else HospitalClient()
    try:
        text = client.export_feedback_jsonl()
    except HospitalClientError as exc:
        print(f"导出失败 [{exc.error_code}] {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    try:
        samples = parse_jsonl(text)
    except ValueError as exc:
        print(f"导出内容不可用: {exc}", file=sys.stderr)
        return 1
    if not samples:
        print("本次没有可导出样本（医院侧没有 LABELED 状态的点踩）")
        return 0

    added = merge_samples(Path(args.output), samples)
    print(f"导出 {len(samples)} 条，新增 {added} 条 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
