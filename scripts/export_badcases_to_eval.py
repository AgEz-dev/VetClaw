"""将 data/badcases.jsonl 中的坏病例脱敏数据一键导出为标准 RAG 评测集样本。

用法:
    python -m scripts.export_badcases_to_eval
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from feedback_store import export_badcases_to_eval_format, DEFAULT_BADCASE_FILE, DEFAULT_EVAL_EXPORT_FILE


def main():
    print(f"🔄 正在从 {DEFAULT_BADCASE_FILE} 导出坏病例样本...")
    res = export_badcases_to_eval_format()
    count = res.get("cases_count", 0)
    print(f"✅ 导出完成！共转化 {count} 条坏病例为评测集用例 -> {DEFAULT_EVAL_EXPORT_FILE}")


if __name__ == "__main__":
    main()
