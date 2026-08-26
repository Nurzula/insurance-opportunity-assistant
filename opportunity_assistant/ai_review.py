"""商机记录的可选 AI 批量复核。

本模块刻意与 Streamlit、pandas 和报告生成逻辑解耦，便于在本地测试及在 UI 中
按需启用。发送给兼容 OpenAI 接口的每条记录默认只包含 ``record_id``、``title``、
``stage``、``amount``、``region`` 五个字段；仅当调用方明确提供时，才额外发送脱敏、
截断后的 ``excerpt`` 与规范化的 ``source_type``。联系人、电话、邮箱和链接不会进入请求。

AI 只作为规则筛选后的辅助复核层。任何配置、网络或模型输出异常都会安全地把对应
记录标记为 ``review``，不会擅自纳入或排除商机。
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal


DEFAULT_BASE_URL: Final = "https://api.deepseek.com"
DEFAULT_MODEL: Final = "deepseek-v4-flash"
DEFAULT_BATCH_SIZE: Final = 30
MAX_BATCH_SIZE: Final = 30
MAX_EXCERPT_LENGTH: Final = 1_800
DEFAULT_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_MAX_ATTEMPTS: Final = 2
DEFAULT_MAX_TOTAL_SECONDS: Final = 180.0

ReviewMode = Literal["insurance", "engineering", "mixed"]

INSURANCE_CATEGORIES: Final[tuple[str, ...]] = (
    "工程险",
    "货运险",
    "意外险",
    "健康险",
    "责任险",
    "企财险",
    "信用险",
    "保证险",
)
ENGINEERING_CATEGORIES: Final[tuple[str, ...]] = ("工程直接", "前期", "无关")
LOCAL_ONLY_CATEGORIES: Final[tuple[str, ...]] = ("待判断",)
DECISIONS: Final[tuple[str, ...]] = ("include", "exclude", "review")

_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "record_id": ("record_id", "记录ID", "记录id", "序号", "id", "ID"),
    "title": ("title", "项目名称", "商机名称", "标题", "招标项目名称"),
    "stage": ("stage", "阶段", "项目阶段", "信息类型", "招标类型", "公告类型"),
    "amount": ("amount", "budget", "金额", "招标金额", "预算金额", "项目金额"),
    "region": ("region", "地区", "区域", "所在地区", "区县", "项目地区"),
    "excerpt": ("excerpt", "公告正文", "内容摘要", "证据摘录"),
    "source_type": ("source_type", "来源类型", "商机类型", "业务类型"),
}

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"(?i)(?<![\w.])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.])")
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9][\d*＊xX]{9}(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)0[\d*＊xX]{2,3}[-—\s]?[\d*＊xX]{7,8}(?!\d)")
_CONTACT_NAME_RE = re.compile(
    r"((?:项目)?联\s*系\s*人\s*[:：]\s*)"
    r"(?:[\u3400-\u9fff·]{1,10}|[A-Za-z][A-Za-z .'-]{0,30})",
    re.IGNORECASE,
)
_LABELED_PHONE_RE = re.compile(
    r"((?:联\s*系\s*)?(?:电\s*话|手\s*机|方\s*式)\s*[:：]\s*)"
    r"[0-9*＊xX\-—()（）\s]{5,30}",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL
)

_MAX_FIELD_LENGTHS: Final[dict[str, int]] = {
    "record_id": 100,
    "title": 500,
    "stage": 120,
    "amount": 100,
    "region": 120,
    "excerpt": MAX_EXCERPT_LENGTH,
    "source_type": 20,
}
_MAX_REASON_LENGTH: Final = 240


class AIReviewProtocolError(ValueError):
    """模型结果不符合本模块的严格 JSON 协议。"""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() in {"", "nan", "NaN", "<NA>", "NaT", "None"}


def _safe_scalar_text(value: Any) -> str:
    """仅接受标量，避免嵌套对象把未授权字段带进提示词。"""

    if _is_missing(value):
        return ""
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return ""
    return str(value).strip()


def _redact_embedded_contacts(text: str) -> str:
    """防止电话、邮箱或链接偶然被粘贴进标题等允许字段。"""

    text = _CONTACT_NAME_RE.sub(r"\1[已移除联系人]", text)
    text = _LABELED_PHONE_RE.sub(r"\1[已移除电话]", text)
    text = _URL_RE.sub("[已移除链接]", text)
    text = _EMAIL_RE.sub("[已移除邮箱]", text)
    text = _MOBILE_RE.sub("[已移除电话]", text)
    text = _LANDLINE_RE.sub("[已移除电话]", text)
    return text


def _first_value(record: Mapping[str, Any], field: str) -> Any:
    for alias in _FIELD_ALIASES[field]:
        if alias in record and not _is_missing(record[alias]):
            return record[alias]
    return ""


def minimize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """返回可发送给模型的最小、脱敏记录。

    缺少 ``record_id`` 时会生成稳定于当前输入顺序的 ``row-N``。重复 ID 会追加
    ``-N`` 后缀，确保模型结果能够无歧义地映射回每一行。
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records 必须是由字典组成的序列。")

    minimized: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"第 {index} 条记录不是字典。")

        item: dict[str, str] = {}
        for field in ("record_id", "title", "stage", "amount", "region"):
            value = _safe_scalar_text(_first_value(record, field))
            value = _redact_embedded_contacts(value)
            item[field] = value[: _MAX_FIELD_LENGTHS[field]]

        excerpt = _safe_scalar_text(_first_value(record, "excerpt"))
        if excerpt:
            item["excerpt"] = _redact_embedded_contacts(excerpt)[
                :MAX_EXCERPT_LENGTH
            ]

        source_type = _normalize_source_type(
            _safe_scalar_text(_first_value(record, "source_type"))
        )
        if source_type:
            item["source_type"] = source_type

        base_id = item["record_id"] or f"row-{index}"
        candidate_id = base_id
        suffix = 2
        while candidate_id in used_ids:
            candidate_id = f"{base_id}-{suffix}"
            suffix += 1
        item["record_id"] = candidate_id[: _MAX_FIELD_LENGTHS["record_id"]]
        # 极端情况下截断可能再次造成重复，使用输入序号保证最终唯一。
        if item["record_id"] in used_ids:
            item["record_id"] = f"row-{index}"
        used_ids.add(item["record_id"])
        minimized.append(item)

    return minimized


def _normalize_source_type(value: str) -> str:
    """仅允许保险/工程标记进入请求，避免把任意来源备注当成模型数据。"""

    aliases = {
        "insurance": "insurance",
        "险": "insurance",
        "保险": "insurance",
        "保险类": "insurance",
        "engineering": "engineering",
        "工程": "engineering",
        "工程类": "engineering",
    }
    return aliases.get(str(value).strip().lower(), "")


def _normalize_review_mode(review_mode: str) -> ReviewMode:
    aliases = {
        "insurance": "insurance",
        "险": "insurance",
        "保险": "insurance",
        "engineering": "engineering",
        "工程": "engineering",
        "mixed": "mixed",
        "混合": "mixed",
    }
    normalized = aliases.get(str(review_mode).strip().lower())
    if normalized is None:
        raise ValueError("review_mode 只能是 insurance、engineering 或 mixed。")
    return normalized  # type: ignore[return-value]


def _system_prompt(review_mode: ReviewMode) -> str:
    common = """
你是保险公司商机筛选的审慎复核员。仅根据输入字段判断，不得补写输入中不存在的事实。
只返回一个严格 JSON 对象，不要 Markdown、代码围栏、解释或前后缀。顶层只能有 reviews。
reviews 必须与输入逐条一一对应、不得增删或重复，record_id 必须原样返回。
每项只能包含 record_id、category、decision、reason、confidence 五个字段：
- decision 只能是 include、exclude、review；证据不足或存在歧义时必须用 review。
- 若记录含 excerpt，必须优先依据 excerpt，再结合标题判断；source_type 仅表示来自保险表或工程表，
  不能替代证据或强迫分类。
- reason 使用简短中文并引用一段来自 excerpt 的短证据；没有 excerpt 时引用标题中的短语。
  不得虚构金额、地区、险种或项目性质，也不得在结果中复述联系人或联系方式。
- confidence 必须是 0 到 1 之间的数字。
""".strip()

    insurance = f"""
当前任务是保险商机复核。category 只能是：{','.join(INSURANCE_CATEGORIES)},无关,待判断。
同一项目同时覆盖多个目标险种时，category 用“、”连接多个标准险种名称，例如“意外险、健康险”。
八类目标险种包括工程保险、货物运输保险、意外伤害保险、健康保险、各类责任保险、
企业财产保险、信用保险、保证保险/履约保证。明确购买上述保险或保证产品才可 include。
保险孔/保险销等机械名词、排危除险、风险评估、社保医保系统、保险公司自身采购装修/软件/
宣传服务、仅在价格条款中出现“保险费”等，均为无关并 exclude。标题不足以确认时 review。
""".strip()

    engineering = """
当前任务是工程商机复核。category 只能是：工程直接,前期,无关,待判断。
工程直接：明确的施工、改造、安装、修缮、基础设施或工程总承包项目。
前期：勘察、设计、规划、可研、造价、监理、检测等工程前期或专业服务线索。
无关：标题中的“工程”只来自机构名称/资质描述，或实际采购为医疗设备、办公用品、车辆、
普通货物、宣传、软件等。明确且符合目标用 include，无关用 exclude，无法确认用 review。
金额只能使用输入值；不得把项目总投资推断为保险保费或采购预算。
""".strip()

    if review_mode == "insurance":
        return f"{common}\n\n{insurance}"
    if review_mode == "engineering":
        return f"{common}\n\n{engineering}"
    return f"{common}\n\n输入为混合商机。分别遵守以下两套口径：\n{insurance}\n\n{engineering}"


def build_review_request(
    records: Sequence[Mapping[str, Any]],
    *,
    review_mode: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """构造一次兼容 OpenAI Chat Completions 的请求参数。

    此函数不联网，可用于 UI 日志审计或单元测试。超过 ``MAX_BATCH_SIZE`` 会拒绝
    构造，调用方应先分批。
    """

    mode = _normalize_review_mode(review_mode)
    minimized = minimize_records(records)
    if not minimized:
        raise ValueError("不能为模型构造空批次。")
    if len(minimized) > MAX_BATCH_SIZE:
        raise ValueError(f"单批最多 {MAX_BATCH_SIZE} 条记录。")

    request: dict[str, Any] = {
        "model": str(model).strip() or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt(mode)},
            {
                "role": "user",
                "content": json.dumps(
                    {"review_mode": mode, "records": minimized},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": min(8_000, 500 + len(minimized) * 180),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if request["model"].lower().startswith("deepseek-v4-"):
        # V4 的结构化分类无需思考链；关闭可降低输出截断和空 content 的概率。
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    return request


def parse_json_object(content: Any) -> dict[str, Any]:
    """从模型 content 中稳健提取一个 JSON 对象。"""

    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        blocks: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                blocks.append(str(part.get("text", "")))
            else:
                blocks.append(str(getattr(part, "text", part)))
        content = "".join(blocks)
    if not isinstance(content, str) or not content.strip():
        raise AIReviewProtocolError("模型返回了空内容。")

    text = content.strip().lstrip("\ufeff")
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise AIReviewProtocolError("模型顶层结果不是 JSON 对象。")
        return value
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise AIReviewProtocolError("模型未返回可解析的 JSON 对象。") from None


def _normalize_category(value: Any, mode: ReviewMode) -> str:
    category = str(value).strip()
    aliases = {
        "直接": "工程直接",
        "直接工程": "工程直接",
        "工程前期": "前期",
        "不相关": "无关",
        "人工复核": "待判断",
        "不确定": "待判断",
    }
    category = aliases.get(category, category)
    if mode == "insurance":
        allowed = set(INSURANCE_CATEGORIES) | {"无关", "待判断"}
        parts = [part.strip() for part in re.split(r"[、,，/]+", category) if part.strip()]
        if len(parts) > 1:
            if any(part not in INSURANCE_CATEGORIES for part in parts):
                raise AIReviewProtocolError(f"未知组合分类：{category}。")
            return "、".join(dict.fromkeys(parts))
    elif mode == "engineering":
        allowed = set(ENGINEERING_CATEGORIES) | {"待判断"}
    else:
        allowed = (
            set(INSURANCE_CATEGORIES)
            | set(ENGINEERING_CATEGORIES)
            | {"待判断"}
        )
    if category not in allowed:
        raise AIReviewProtocolError(f"未知分类：{category or '空值'}。")
    return category


def _normalize_decision(value: Any) -> str:
    decision = str(value).strip().lower()
    aliases = {
        "纳入": "include",
        "保留": "include",
        "排除": "exclude",
        "剔除": "exclude",
        "复核": "review",
        "待复核": "review",
    }
    decision = aliases.get(decision, decision)
    if decision not in DECISIONS:
        raise AIReviewProtocolError(f"未知处理决定：{decision or '空值'}。")
    return decision


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise AIReviewProtocolError("confidence 不能是布尔值。")
    text = str(value).strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        confidence = float(text)
    except (TypeError, ValueError):
        raise AIReviewProtocolError("confidence 不是数字。") from None
    if is_percent or 1 < confidence <= 100:
        confidence /= 100
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise AIReviewProtocolError("confidence 必须介于 0 和 1。")
    return round(confidence, 4)


def validate_review_object(
    payload: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]],
    *,
    review_mode: str,
) -> list[dict[str, Any]]:
    """严格校验模型对象，并按输入顺序返回标准化结果。"""

    mode = _normalize_review_mode(review_mode)
    expected = minimize_records(expected_records)
    if set(payload.keys()) != {"reviews"}:
        raise AIReviewProtocolError("JSON 顶层只能包含 reviews。")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise AIReviewProtocolError("reviews 必须是数组。")
    if len(reviews) != len(expected):
        raise AIReviewProtocolError("模型结果数量与输入记录数量不一致。")

    required_fields = {"record_id", "category", "decision", "reason", "confidence"}
    by_id: dict[str, dict[str, Any]] = {}
    expected_ids = {item["record_id"] for item in expected}
    for index, item in enumerate(reviews, start=1):
        if not isinstance(item, Mapping) or set(item.keys()) != required_fields:
            raise AIReviewProtocolError(f"第 {index} 条结果字段不符合协议。")
        record_id = str(item["record_id"]).strip()
        if record_id not in expected_ids or record_id in by_id:
            raise AIReviewProtocolError("模型返回了未知或重复的 record_id。")
        reason = _redact_embedded_contacts(str(item["reason"]).strip())
        if not reason:
            raise AIReviewProtocolError("reason 不能为空。")
        normalized = {
            "record_id": record_id,
            "category": _normalize_category(item["category"], mode),
            "decision": _normalize_decision(item["decision"]),
            "reason": reason[:_MAX_REASON_LENGTH],
            "confidence": _normalize_confidence(item["confidence"]),
            "source": "ai",
        }
        if normalized["category"] == "待判断":
            normalized["decision"] = "review"
        if normalized["category"] == "无关" and normalized["decision"] == "include":
            raise AIReviewProtocolError("无关记录不能标记为 include。")
        by_id[record_id] = normalized

    return [by_id[item["record_id"]] for item in expected]


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_response_content(response: Any) -> Any:
    choices = _get_value(response, "choices", None)
    if not choices:
        raise AIReviewProtocolError("模型没有返回候选结果。")
    choice = choices[0]
    finish_reason = _get_value(choice, "finish_reason", None)
    if finish_reason not in (None, "stop"):
        raise AIReviewProtocolError(f"模型响应未正常结束（{finish_reason}）。")
    message = _get_value(choice, "message", None)
    content = _get_value(message, "content", None)
    if content is None:
        raise AIReviewProtocolError("模型返回了空内容。")
    return content


def _fallback_reviews(
    records: Sequence[Mapping[str, Any]], reason: str
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": item["record_id"],
            "category": "待判断",
            "decision": "review",
            "reason": reason[:_MAX_REASON_LENGTH],
            "confidence": 0.0,
            "source": "fallback",
        }
        for item in minimize_records(records)
    ]


def _friendly_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "401" in text or "unauthorized" in text or "invalid api key" in text:
        return "AI 密钥无效或没有所选模型权限，已转入人工复核。"
    if "429" in text or "rate" in text or "余额" in text or "quota" in text:
        return "AI 接口限流或余额不足，已转入人工复核。"
    if "timeout" in text or "timed out" in text or "超时" in text:
        return "AI 请求超时，已转入人工复核。"
    if isinstance(exc, (AIReviewProtocolError, json.JSONDecodeError)):
        return "AI 未返回符合结构化协议的结果，已转入人工复核。"
    return "AI 复核暂时不可用，已转入人工复核。"


def _is_fatal_provider_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "401",
            "unauthorized",
            "invalid api key",
            "insufficient",
            "余额不足",
            "quota exceeded",
        )
    )


def _default_client_factory(
    *, api_key: str, base_url: str, timeout: float
) -> Any:
    # 延迟导入：没有密钥时模块不会创建客户端，也不会触发任何网络行为。
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def _base_result(
    *,
    status: str,
    message: str,
    records: Sequence[Mapping[str, Any]],
    reviews: list[dict[str, Any]],
    model: str,
    base_url: str,
    batch_size: int,
    batch_count: int,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reviewed_count = sum(item.get("source") == "ai" for item in reviews)
    return {
        "enabled": status not in {"disabled", "empty", "config_error"},
        "ok": status in {"ok", "empty"},
        "status": status,
        "message": message,
        "reviews": reviews,
        "submitted_count": len(records),
        "reviewed_count": reviewed_count,
        "fallback_count": len(reviews) - reviewed_count,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "model": model,
        "base_url": base_url,
        "errors": errors or [],
    }


def review_opportunities(
    records: Sequence[Mapping[str, Any]],
    api_key: str | None,
    *,
    review_mode: str = "mixed",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
    client: Any | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """批量复核商机并返回永不泄露密钥的友好状态对象。

    ``client`` 和 ``client_factory`` 用于离线测试或接入其他 OpenAI 兼容客户端。
    一次模型调用最多 ``MAX_BATCH_SIZE`` 条，更多记录会自动拆成多个有界批次。
    """

    cleaned_base_url = str(base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    cleaned_model = str(model or DEFAULT_MODEL).strip()
    try:
        mode = _normalize_review_mode(review_mode)
        effective_batch_size = min(int(batch_size), MAX_BATCH_SIZE)
        if effective_batch_size < 1:
            raise ValueError("batch_size 必须大于 0。")
        attempts = min(max(int(max_attempts), 1), 3)
        cleaned_records = minimize_records(records)
        total_seconds = max(30.0, min(float(max_total_seconds), 600.0))
    except (TypeError, ValueError) as exc:
        return _base_result(
            status="config_error",
            message=f"AI 复核配置无效：{exc}",
            records=[],
            reviews=[],
            model=cleaned_model,
            base_url=cleaned_base_url,
            batch_size=0,
            batch_count=0,
        )

    if not cleaned_records:
        return _base_result(
            status="empty",
            message="没有需要 AI 复核的记录。",
            records=cleaned_records,
            reviews=[],
            model=cleaned_model,
            base_url=cleaned_base_url,
            batch_size=effective_batch_size,
            batch_count=0,
        )

    if not str(api_key or "").strip():
        reviews = _fallback_reviews(
            cleaned_records, "AI 复核未启用：未配置 API Key，请人工复核。"
        )
        return _base_result(
            status="disabled",
            message="未配置 API Key，AI 复核已安全跳过。",
            records=cleaned_records,
            reviews=reviews,
            model=cleaned_model,
            base_url=cleaned_base_url,
            batch_size=effective_batch_size,
            batch_count=0,
        )

    if client is None:
        factory = client_factory or _default_client_factory
        try:
            client = factory(
                api_key=str(api_key).strip(),
                base_url=cleaned_base_url,
                timeout=float(timeout),
            )
        except Exception as exc:  # 客户端实现和兼容服务可能抛出不同异常类型。
            reason = _friendly_error(exc)
            reviews = _fallback_reviews(cleaned_records, reason)
            return _base_result(
                status="error",
                message=reason,
                records=cleaned_records,
                reviews=reviews,
                model=cleaned_model,
                base_url=cleaned_base_url,
                batch_size=effective_batch_size,
                batch_count=0,
                errors=[{"batch": 0, "error_type": type(exc).__name__, "message": reason}],
            )

    all_reviews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    batches = [
        cleaned_records[index : index + effective_batch_size]
        for index in range(0, len(cleaned_records), effective_batch_size)
    ]

    started = time.monotonic()
    processed_batch_count = 0
    for batch_number, batch in enumerate(batches, start=1):
        if time.monotonic() - started >= total_seconds:
            remaining = [item for later in batches[batch_number - 1 :] for item in later]
            reason = "AI 批量复核达到全局时限，剩余记录已转入人工复核。"
            all_reviews.extend(_fallback_reviews(remaining, reason))
            errors.append(
                {"batch": batch_number, "error_type": "GlobalTimeout", "message": reason}
            )
            break
        processed_batch_count += 1
        last_error: Exception | None = None
        for _attempt in range(1, attempts + 1):
            try:
                request = build_review_request(
                    batch, review_mode=mode, model=cleaned_model
                )
                response = client.chat.completions.create(**request)
                payload = parse_json_object(_extract_response_content(response))
                batch_reviews = validate_review_object(
                    payload, batch, review_mode=mode
                )
                all_reviews.extend(batch_reviews)
                last_error = None
                break
            except Exception as exc:  # 安全回退优先，不能让模型问题阻断主流程。
                last_error = exc
                if _is_fatal_provider_error(exc):
                    break
        if last_error is not None:
            reason = _friendly_error(last_error)
            all_reviews.extend(_fallback_reviews(batch, reason))
            errors.append(
                {
                    "batch": batch_number,
                    "error_type": type(last_error).__name__,
                    "message": reason,
                }
            )
            if _is_fatal_provider_error(last_error):
                remaining = [item for later in batches[batch_number:] for item in later]
                if remaining:
                    all_reviews.extend(_fallback_reviews(remaining, reason))
                break

    reviewed_count = sum(item["source"] == "ai" for item in all_reviews)
    if reviewed_count == len(cleaned_records):
        status = "ok"
        message = f"AI 复核完成，共处理 {reviewed_count} 条记录。"
    elif reviewed_count:
        status = "partial"
        message = (
            f"AI 复核部分完成：成功 {reviewed_count} 条，"
            f"转人工复核 {len(cleaned_records) - reviewed_count} 条。"
        )
    else:
        status = "error"
        message = "AI 复核未完成，全部记录已安全转入人工复核。"

    return _base_result(
        status=status,
        message=message,
        records=cleaned_records,
        reviews=all_reviews,
        model=cleaned_model,
        base_url=cleaned_base_url,
        batch_size=effective_batch_size,
        batch_count=processed_batch_count,
        errors=errors,
    )


__all__ = [
    "AIReviewProtocolError",
    "DEFAULT_BASE_URL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MODEL",
    "ENGINEERING_CATEGORIES",
    "INSURANCE_CATEGORIES",
    "MAX_BATCH_SIZE",
    "MAX_EXCERPT_LENGTH",
    "build_review_request",
    "minimize_records",
    "parse_json_object",
    "review_opportunities",
    "validate_review_object",
]
