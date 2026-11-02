"""Agent 工具集（第二批）：本地药物规则 2 个 + 医院远端 4 个，合计 7 个（含既有检索）。

统一返回口径
------------
除既有的 ``search_knowledge_base``（字符串格式已被引用状态机依赖）外，本模块工具
一律返回 JSON 字符串：``{"ok", "data", "error_code", "elapsed_ms"}``（失败时附
``message``）。**失败也返回结构化结果而不是抛异常**——异常会被 ``execute_tool``
转成普通字符串，模型与轨迹都无法区分"工具没跑"与"工具跑了但失败"。

安全边界
--------
- 远端工具只读三条 + 唯一写「转人工」；药品规则工具**只给参考**，医院侧发药闸门
  仍是唯一裁决者（LLM 不接管安全判定）。
- 医院共享密钥为空时远端工具直接失败（``HOSPITAL_NOT_CONFIGURED``），不尝试匿名访问。
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from datetime import date

from case_context import get_case_id
from hospital_client import HospitalClientError, hospital_provider
from prescription_review import PrescriptionReviewer
from tools import registry

logger = logging.getLogger(__name__)

_reviewer: PrescriptionReviewer | None = None


class ToolError(Exception):
    """工具可预期的业务性失败（带稳定错误码），与程序异常区分。"""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


def _get_reviewer() -> PrescriptionReviewer:
    global _reviewer
    if _reviewer is None:
        _reviewer = PrescriptionReviewer()
    return _reviewer


def _envelope(ok: bool, data, started: float, error_code: str | None = None,
              message: str | None = None) -> str:
    body = {
        "ok": ok,
        "data": data,
        "error_code": error_code,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
    if message:
        body["message"] = message
    return json.dumps(body, ensure_ascii=False)


def _tool(fn):
    """把执行结果/异常统一收敛进 envelope；失败绝不冒泡。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        try:
            return _envelope(True, fn(*args, **kwargs), started)
        except (ToolError, HospitalClientError) as exc:
            return _envelope(False, None, started, exc.error_code, str(exc))
        except Exception as exc:  # 工具层兜底：模型应看到"失败"，而不是把回合炸掉
            logger.exception("tool error: %s", fn.__name__)
            return _envelope(False, None, started, "TOOL_ERROR",
                             f"{type(exc).__name__}: {exc}")

    return wrapper


# --------------------------------------------------------------------- 本地：药品规则

@_tool
def lookup_drug_label(drug_name: str) -> dict:
    """查询已收录药品说明书要点（别名/禁用物种/参考剂量/频次上限/引用）；drug_name 可为商品名或通用名。"""
    reviewer = _get_reviewer()
    drug = reviewer.find_drug(drug_name)
    if drug is None:
        raise ToolError("DRUG_NOT_IN_TABLE",
                        f"「{drug_name}」不在已收录的说明书范围内，无法核验其禁忌与剂量")
    return {
        "label": drug.get("label"),
        "matched_by": [n for n in drug.get("names", [])
                       if str(n).lower() in str(drug_name).lower()],
        "species_banned": drug.get("species_banned", []),
        "dose_mg_per_kg": drug.get("dose_mg_per_kg", {}),
        "fixed_dose_species": drug.get("fixed_dose_species", []),
        "max_frequency_per_day": drug.get("max_frequency_per_day", {}),
        "min_age_weeks": drug.get("min_age_weeks"),
        "min_weight_kg": drug.get("min_weight_kg"),
        "notes": drug.get("notes"),
        "citations": drug.get("citations", {}),
    }


@_tool
def check_drug_safety(drug_name: str, species: str = "", allergies: str = "",
                      weight_kg: float = None, age_weeks: float = None,
                      dose_mg_per_kg: float = None) -> dict:
    """按规则表核对单个药品的安全性（过敏/物种禁用/剂量/年龄体重），返回 verdict 与逐条理由；结论仅供医生参考。"""
    patient = {"species": species, "allergies": allergies,
               "weight_kg": weight_kg, "age_weeks": age_weeks}
    item = {"drug_name": drug_name, "dose_mg_per_kg": dose_mg_per_kg}
    return _get_reviewer().review({
        "patient": {k: v for k, v in patient.items() if v not in (None, "")},
        "prescription": {"items": [item]},
    })


# --------------------------------------------------------------------- 远端：医院数据

@_tool
def get_pet_profile(pet_id: int) -> dict:
    """查询宠物档案（物种/品种/性别/出生日期/体重/绝育/过敏史/认证状态）；pet_id 取系统提示里的「档案ID」。"""
    return hospital_provider.get().get_pet_profile(pet_id)


@_tool
def get_vaccination_history(pet_id: int) -> list:
    """查询宠物接种史（疫苗名/剂次/接种日期/下次应接种日期/来源），按接种日期倒序。"""
    return hospital_provider.get().get_vaccinations(pet_id)


@_tool
def get_medication_history(pet_id: int) -> list:
    """查询宠物既往用药史（仅已发药处方：发药时间/医嘱/药品明细），按发药时间倒序。"""
    return hospital_provider.get().get_medications(pet_id)


@_tool
def escalate_to_human(pet_id: int, reason: str) -> dict:
    """转人工：把当前病例交给医院医生跟进（唯一写操作）；reason 必填，写明为什么需要医生介入。"""
    clean_reason = str(reason or "").strip()
    if not clean_reason:
        raise ValueError("转人工必须说明原因")
    # 幂等键包含 case：同一宠物同一天、同一原因但不同就诊，必须是两张工单，
    # 否则医院侧会因「同键异指纹」直接 409。
    case_id = get_case_id()
    key = hashlib.sha256(
        f"{date.today().isoformat()}|{pet_id}|{case_id}|{clean_reason}".encode("utf-8")
    ).hexdigest()[:32]
    return hospital_provider.get().escalate(
        pet_id=pet_id, case_id=case_id, reason=clean_reason,
        idempotency_key=key, source="VETCLAW")


# --------------------------------------------------------------------- 注册

def register_to(reg):
    for fn in (lookup_drug_label, check_drug_safety,
               get_pet_profile, get_vaccination_history, get_medication_history,
               escalate_to_human):
        reg.register(fn)


register_to(registry)
