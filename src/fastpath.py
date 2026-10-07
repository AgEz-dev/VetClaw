"""Fast-Path 规则守卫：入口前置拦截 P0 急症与未知处方药。

- emergency: 物种 + 剧毒成分 + 动作词（排除咨询句式）→ ChromaDB 点查 SOP + 毒物 chunk 拼接
- refuse: 剂量诉求 + watchlist 处方药 + 不在白名单 → 分诊引导
- pass: 其他 → 正常走 RAG/Agent
"""
import json
import re
from pathlib import Path


class FastPathGuard:
    def __init__(self, rules_path="rules/fastpath_rules.json",
                 rules_dict=None, pipeline=None):
        if rules_dict is not None:
            self.rules = rules_dict
        else:
            self.rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self._pipeline = pipeline

    def check(self, query: str) -> dict:
        q = query.lower()

        # 规则 1：P0 急症
        if not re.search(self.rules["consult_pattern"], q):
            for toxin in self.rules["p0_toxins"]:
                if not any(k.lower() in q for k in toxin["keywords"]):
                    continue
                if not any(s in query for s in toxin["species"]):
                    continue
                if re.search(self.rules["action_regex"], query):
                    return {"action": "emergency", "toxin": toxin["name"],
                            "species": [s for s in toxin["species"] if s in query][0],
                            "chunk_id": toxin["chunk_id"]}

        # 规则 2：未知处方药
        if any(d in query for d in self.rules["dosage_words"]):
            if not any(w in q for w in self.rules["drug_whitelist"]):
                for drug in self.rules["unlisted_drugs_watchlist"]:
                    if drug in query:
                        return {"action": "refuse", "drug_hint": drug}

        return {"action": "pass"}

    def emergency_message(self, toxin: str, species: str, chunk_id: str) -> str:
        """从 ChromaDB 点查 SOP chunk + 毒物 chunk 拼接（<3ms，不走 embedding）。"""
        sop = self._fetch(self.rules["sop_chunk_id"])
        detail = self._fetch(chunk_id)
        parts = [f"【P0 急症预警】检测到{species}可能接触{toxin}。"]
        if sop:
            parts.append(sop)
        if detail:
            parts.append(f"---\n关于{toxin}的详细信息：\n{detail}")
        parts.append("\n⚠️ 以上建议仅供参考，不能替代执业兽医诊断，紧急情况请立即就医。")
        return "\n\n".join(parts)

    def refuse_message(self, drug_hint: str) -> str:
        return (
            f"【分诊引导】{drug_hint}不在我已收录的官方说明书范围内，我不会凭空推测用药剂量。"
            f"请先做现场排查：1) 观察宠物牙龈颜色（粉红/发白/发青）与静息呼吸频率；"
            f"2) 核对药盒成分表是否含对乙酰氨基酚、木糖醇、葱属精油等已知剧毒成分；"
            f"3) 记录误食时间与估计剂量。请携带原药盒与上述体征数据尽快前往宠物医院急诊。"
        )

    def _fetch(self, chunk_id: str) -> str | None:
        if self._pipeline is None:
            return None
        res = self._pipeline.collection.get(ids=[chunk_id])
        if res["documents"]:
            return res["documents"][0]
        return None
