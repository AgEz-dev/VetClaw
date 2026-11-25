"""预问诊会话状态机：把「槽位引擎 + LLM 通道」串成一轮一轮可驱动的流程。

职责边界：
- **本模块**负责流程编排（开场 → 追问 → 抽取 → 收尾）与会话内状态；
- **落库、鉴权、审计**由 Java 侧负责（VetClaw 无状态，契约 §3.4）；
- **急症拦截**复用既有 ``FastPathGuard``（不重复实现安全逻辑）。

设计要点：
- ``llm`` 是可注入的 ``callable(messages) -> str``。**不传则退化为纯规则模式**（无网络也能跑），
  这样单测、离线环境、模型不可用时都有明确行为，而不是崩掉。
- 每轮最多问 N 个（模板 ``meta.rules``），自由文本问题独占一轮；
- 收尾：必填齐备 或 达轮数上限（后者把未采集项**如实**写进 ``incomplete``）；
- 急症命中：**立即中断**（不再追问），摘要照出并标记 ``interrupted``。
"""
from __future__ import annotations

from intake import (
    load_template, empty_state, merge_archive, apply_reply, next_questions,
    is_complete, incomplete_keys, render_status,
)
from intake_llm import (
    slots_without_options, merge_llm_slots, parse_extract_response,
    build_extract_messages, build_summary_messages, parse_summary_response,
)

OPENING = (
    "您好，我是宠安医院的**预问诊助手**。医生会在诊前看到我帮您整理的情况 —— "
    "这样您到院后能直接聊重点。\n\n"
    "您**随便说、不用组织语言**，我会帮您整理成医生看得懂的病史。\n\n"
    "**可以像这样说：**\n"
    "「它从昨晚开始一直吐，三四次吧，精神也没以前好，没乱吃东西，我们还没给它吃药」\n\n"
    "我的整理只帮医生提前准备，**不构成诊断**；过程中您随时可以补充或纠正。"
)

_EMERGENCY_SUMMARY = "【急症预警】宠主描述命中急症规则，预问诊已中断，建议立即线下就医。"


def _as_items(picks: list[dict]) -> list[dict]:
    """追问项 → 结构化返回（``key`` 供调用方驱动门控与展示，``ask`` 供展示）。"""
    return [{"key": q["key"], "name": q.get("name"), "ask": q.get("ask"),
             "options": q.get("options")} for q in picks]


def build_summary_fallback(state: dict, template: dict, conversation_text: str = "") -> dict:
    """**无 LLM 时的摘要兜底**：直接从槽位值拼出 SOAP 骨架（不加工、不推断）。

    存在的意义有两个：模型不可用时流程仍能交付；同时它是**摘要质量的下界基线**
    —— LLM 版应当明显好于它，否则说明提示词有问题。
    """
    status = {row["key"]: row for row in render_status(state, template, conversation_text)}

    def val(key: str) -> str:
        row = status.get(key, {})
        return row.get("value", "") if row.get("state") != "missing" else ""

    present = {
        "onset": val("onset"),
        "characteristics": "；".join(x for x in (val("vomit_content"), val("vomit_relation"),
                                                 val("vomit_kind"), val("vomit_freq")) if x),
        "progression": val("progression"),
        "associated": val("vomit_associated"),
        "treatment": val("treatment"),
    }
    past = {"chronic": val("past_disease"), "surgery": "",
            "vaccination": val("vaccine"), "allergy": val("allergy")}
    general = {"appetite": "", "water": "", "weight_change": ""}

    findings = []
    free_note = val("free_note")
    other = val("other_findings")
    if free_note:
        findings.append({"text": free_note, "source": "free_text"})
    if other:
        findings.append({"text": other, "source": "free_text"})

    return {
        # ⚠️ 主诉 = 症状 + **持续时间**。不能把 onset（急缓，如"慢慢变严重"）拼进来 ——
        # 实测拼出来的"呕吐 慢慢变严重"不像主诉（急缓归 present_illness.onset）。
        "chief_complaint": val("chief"),
        "present_illness": present,
        "past_history": past,
        "general": general,
        "other_findings": findings,
        "ai_hints": [],                     # 无模型时不给提示（不编造）
        "suggested_questions": [],
    }


def detect_memory_conflict(current_text_or_slots: dict | str, l3_profile: dict | None) -> list[dict]:
    """检测当前预问诊输入与历史 L3 长期档案的高危冲突 (过敏史、慢性器官衰竭、禁药)。"""
    if not l3_profile:
        return []

    text = ""
    if isinstance(current_text_or_slots, str):
        text = current_text_or_slots
    elif isinstance(current_text_or_slots, dict):
        text = " ".join(str(v) for v in current_text_or_slots.values() if v)

    conflicts = []
    allergies = str(l3_profile.get("allergies") or "")
    chronic = str(l3_profile.get("chronic_diseases") or l3_profile.get("chronic") or "")

    # 1. 过敏史冲突 (青霉素/阿莫西林/头孢)
    if any(k in allergies for k in ["青霉素", "阿莫西林", "头孢", "磺胺"]):
        if any(tr in text for tr in ["感染", "发炎", "脓", "消炎药", "抗生素", "打针", "青霉素", "阿莫西林", "头孢", "细菌"]):
            conflicts.append({
                "type": "ALLERGY_CONFLICT",
                "risk_title": "[历史档案风险提示 · 药物过敏红线]",
                "history_fact": f"既往过敏史: {allergies}",
                "high_priority_question": f"【历史档案风险提示】档案记录患宠既往存在「{allergies}」高危过敏史。请问此前过敏曾出现哪些具体反应（如面部水肿/皮疹/休克）？当前是否已私自喂食过消炎药？",
                "doctor_alert": f"⚠️ 【历史档案过敏警示】患宠记录有「{allergies}」过敏史，处方抗感染严禁开具相关致敏药物！"
            })

    # 2. 慢性肾病冲突 (呕吐脱水极易加剧肾衰)
    if any(k in chronic for k in ["肾病", "肾衰", "肾功能不全", "CKD"]):
        if any(tr in text for tr in ["吐", "呕吐", "不吃", "脱水", "少尿", "无尿"]):
            conflicts.append({
                "type": "CHRONIC_RENAL_CONFLICT",
                "risk_title": "[历史档案风险提示 · 慢性肾病急性恶化防线]",
                "history_fact": f"既往慢性病: {chronic}",
                "high_priority_question": "【历史档案风险提示】档案记录患宠患有慢性肾功能不全，频繁呕吐脱水极易诱发急性肾前性肾衰竭。请问最近是否出现完全不喝水或尿量显著减少？",
                "doctor_alert": "⚠️ 【历史档案重症警示】基础病慢性肾衰合并急性呕吐脱水，到院需优先建立静脉通道并加急化验肾功生化！"
            })

    # 3. 基础心脏病冲突 (咳嗽气喘输液红线)
    if any(k in chronic for k in ["心脏", "心肌病", "心衰", "HCM"]):
        if any(tr in text for tr in ["咳", "喘", "呼吸急促", "张嘴呼吸", "发喘"]):
            conflicts.append({
                "type": "HEART_FAILURE_CONFLICT",
                "risk_title": "[历史档案风险提示 · 心源性呼吸窘迫防线]",
                "history_fact": f"既往慢性病: {chronic}",
                "high_priority_question": "【历史档案风险提示】档案记录患宠有基础心脏病史，当前咳嗽气喘需高度警惕心源性肺水肿。请问爱宠在安静睡觉时的呼吸次数（静息呼吸率）每分钟是否超过 30 次？",
                "doctor_alert": "⚠️ 【历史档案急症警示】基础心脏病史合并呼吸困难，切忌过量/快速补液，到院严防心源性肺水肿并发症！"
            })

    return conflicts


class IntakeSession:
    """一次预问诊会话（内存态；同一用户 + 同一预约对应一个实例）。"""

    def __init__(self, template: dict | None = None, llm=None, archive: dict | None = None,
                 guard=None, patient_context: dict | None = None, max_turns: int | None = None,
                 l3_profile: dict | None = None):
        self.template = template or load_template()
        self.llm = llm                      # callable(messages) -> str；None = 纯规则模式
        self.guard = guard                  # FastPathGuard；None = 不做急症中断
        self.patient_context = patient_context or {}
        self.l3_profile = l3_profile or (self.patient_context.get("l3_profile") if isinstance(self.patient_context, dict) else {}) or {}
        self.max_turns = max_turns or self.template["meta"]["rules"]["max_turns"]
        self.scope = slots_without_options(self.template)

        self.state = merge_archive(empty_state(self.template, ""), archive, self.template)
        self.conversation = ""
        self.turns = 0
        self.interrupted = False
        self.emergency = None
        self.summary: dict | None = None
        self._asked: list[str] = []
        self.detected_conflicts: list[dict] = []
        self._memory_conflict_asked: bool = False

    # ------------------------------------------------------------------ 状态续传

    def to_state(self) -> dict:
        """导出**可续传的不透明状态**，由调用方（Java）保存并在下次请求原样带回。"""
        return {
            "slots": self.state,
            "conversation": self.conversation,
            "turns": self.turns,
            "asked": list(self._asked),
            "interrupted": self.interrupted,
            "emergency": self.emergency,
            "detected_conflicts": list(self.detected_conflicts),
            "memory_conflict_asked": self._memory_conflict_asked,
        }

    @classmethod
    def from_state(cls, state: dict | None, **kwargs) -> "IntakeSession":
        """从续传状态重建会话（首次调用传 ``None`` 即为全新会话）。"""
        session = cls(**kwargs)
        if not state:
            return session
        for key, value in (state.get("slots") or {}).items():
            session.state[key] = value
        session.conversation = state.get("conversation", "") or ""
        session.turns = int(state.get("turns") or 0)
        session._asked = list(state.get("asked") or [])
        session.interrupted = bool(state.get("interrupted"))
        session.emergency = state.get("emergency")
        session.detected_conflicts = list(state.get("detected_conflicts") or [])
        session._memory_conflict_asked = bool(state.get("memory_conflict_asked"))
        return session

    # ------------------------------------------------------------------ 对外接口

    def opening(self) -> dict:
        """开场消息 + 第一批问题（真实场景里由前端一起展示）。

        ``questions`` 是**结构化**的（含 ``key``）：调用方既要知道问什么（展示 ``ask``），
        也要知道**问的是哪个槽位** —— 抽取的"问过才启用同义词/自由判据"门控依赖它。
        只返回问句文本会让调用方无法正确驱动门控。
        """
        picks = next_questions(self.state, self.template, "")
        self._asked = [q["key"] for q in picks]
        return {
            "mode": "intake",
            "message": OPENING,
            "questions": _as_items(picks),
            "slots": render_status(self.state, self.template, ""),
            "state": self.to_state(),
        }

    def handle(self, reply: str) -> dict:
        """处理宠主的一次回复，返回下一步（继续问 / 出摘要）。"""
        self.conversation += (reply or "") + "\n"

        # ① 急症优先：命中即中断追问（安全 ≥ 完整）
        hit = self._check_emergency(reply)
        if hit:
            self.interrupted = True
            self.emergency = hit
            return self._finish()

        # ② 历史 L3 长期记忆冲突主动召回与高优打断
        conflicts = detect_memory_conflict(reply, self.l3_profile)
        if conflicts and not self._memory_conflict_asked:
            self._memory_conflict_asked = True
            self.detected_conflicts.extend(conflicts)
            c = conflicts[0]
            preempt_picks = [{
                "key": "memory_conflict_preemption",
                "name": "高危档案冲突追问",
                "ask": c["high_priority_question"],
                "options": [
                    "未私自喂药，此前仅轻度皮疹/胃肠不适",
                    "曾出现呼吸急促/过敏性休克高危",
                    "已私自喂食药物（请在下方说明药名与剂量）"
                ]
            }]
            self._asked = [q["key"] for q in preempt_picks]
            self.turns += 1
            return {
                "mode": "intake",
                "done": False,
                "questions": _as_items(preempt_picks),
                "slots": render_status(self.state, self.template, self.conversation),
                "state": self.to_state(),
            }

        # ③ 规则通道（选项型）→ ④ LLM 通道（自由文本，仅补缺）
        apply_reply(self.state, reply, self.template, self.conversation,
                    archive=None, asked_keys=self._asked)
        if self.llm is not None:
            try:
                raw = self.llm(build_extract_messages(self.conversation, self.template,
                                                      focus_keys=self.scope))
                merge_llm_slots(self.state, parse_extract_response(raw, self.template))
            except Exception:
                pass            # 模型不可用/超时**不得阻断预问诊** —— 规则通道已提供兜底

        self.turns += 1
        if is_complete(self.state, self.template, self.conversation) or self.turns >= self.max_turns:
            return self._finish()

        picks = next_questions(self.state, self.template, self.conversation)
        self._asked = [q["key"] for q in picks]
        return {
            "mode": "intake",
            "done": False,
            "questions": _as_items(picks),
            "slots": render_status(self.state, self.template, self.conversation),
            "state": self.to_state(),
        }

    def snapshot(self) -> dict:
        """契约 §3.4 的 ``done.intake`` 载荷。"""
        current_sum = self.summary or self._build_summary()
        return {
            "slots": {row["key"]: {"state": row["state"], "value": row["value"],
                                   "source": row["source"]}
                      for row in render_status(self.state, self.template, self.conversation)},
            "summary": current_sum,
            "ai_hints": (current_sum or {}).get("ai_hints", []),
            "suggested_questions": (current_sum or {}).get("suggested_questions", []),
            "free_text": self.state.get("free_note", {}).get("value", ""),
            "incomplete": incomplete_keys(self.state, self.template, self.conversation),
            "interrupted": self.interrupted,
            "emergency": self.emergency,
        }

    # ------------------------------------------------------------------ 内部

    def _check_emergency(self, reply: str):
        if self.guard is None or not reply:
            return None
        try:
            hit = self.guard.check(reply, self.patient_context)
        except Exception:
            return None                     # 守卫异常不得阻断预问诊
        return hit if hit.get("action") in ("emergency", "refuse") else None

    def _finish(self) -> dict:
        self.summary = self._build_summary()
        return {
            "mode": "intake",
            "done": True,
            "questions": [],
            "slots": render_status(self.state, self.template, self.conversation),
            "intake": self.snapshot(),
            "state": self.to_state(),
        }

    def _build_summary(self) -> dict:
        if self.interrupted and self.emergency:
            fallback = build_summary_fallback(self.state, self.template, self.conversation)
            fallback["ai_hints"] = [{"text": _EMERGENCY_SUMMARY, "citations": []}]
            self._inject_conflict_hints(fallback)
            return fallback
        if self.llm is None:
            fallback = build_summary_fallback(self.state, self.template, self.conversation)
            self._inject_conflict_hints(fallback)
            return fallback
        try:
            raw = self.llm(build_summary_messages(
                self.conversation, render_status(self.state, self.template, self.conversation)))
            summary = parse_summary_response(raw)
        except Exception:
            fallback = build_summary_fallback(self.state, self.template, self.conversation)
            self._inject_conflict_hints(fallback)
            return fallback
        if not summary.get("chief_complaint") and not any(summary.get("present_illness", {}).values()):
            fallback = build_summary_fallback(self.state, self.template, self.conversation)
            self._inject_conflict_hints(fallback)
            return fallback
        self._inject_conflict_hints(summary)
        return summary

    def _inject_conflict_hints(self, summary_dict: dict) -> None:
        if not self.detected_conflicts:
            return
        hints = summary_dict.setdefault("ai_hints", [])
        for c in self.detected_conflicts:
            doctor_alert = c.get("doctor_alert")
            if doctor_alert and not any(h.get("text") == doctor_alert for h in hints):
                hints.insert(0, {"text": doctor_alert, "citations": []})
