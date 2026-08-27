"""商机公告关键详情的确定性提取器。

本模块只处理调用方已经取得的项目行和公告文字，不访问网络，也不调用大模型。
提取原则是“宁可留空，不做猜测”：优先使用项目行中的明确字段，其次仅从带标签
的正文段落或标题中出现的采购方式短语提取信息。所有面向展示的长文本均有硬性
长度上限，避免一条公告撑坏 Excel 或企业微信页面。

公开接口：

* :func:`extract_opportunity_details`：从一行字典/Series 返回详情字典；
* :func:`enrich_opportunity_details`：返回补齐详情字段的 DataFrame 副本。
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import html
import math
import re
from typing import Any, Mapping, Sequence


DETAIL_FIELD_LIMITS: Mapping[str, int] = {
    "project_number": 80,
    "procurement_method": 40,
    "project_location": 120,
    "project_scope": 480,
    "service_term": 320,
    "qualification": 700,
    "key_points": 500,
    "detail_status": 40,
    "detail_source_url": 500,
}

DETAIL_FIELDS: tuple[str, ...] = tuple(DETAIL_FIELD_LIMITS)


_EMPTY_MARKERS = frozenset(
    {
        "",
        "-",
        "--",
        "—",
        "/",
        "无",
        "暂无",
        "未提供",
        "未披露",
        "不详",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
)

_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|li|tr|table|section|article|h[1-6]|ul|ol)[^>]*>",
    flags=re.IGNORECASE,
)
_BREAK_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]{0,500}>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HAN_HEADING = "一二三四五六七八九十百"


_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "project_number": (
        "project_number",
        "项目编号",
        "采购项目编号",
        "采购编号",
        "招标编号",
        "询价书编号",
        "询价通知书编码",
        "项目代码",
        "标段编号",
    ),
    "procurement_method": (
        "procurement_method",
        "采购方式",
        "招标方式",
        "比选方式",
        "询价方式",
    ),
    "project_location": (
        "project_location",
        "项目地点",
        "建设地点",
        "服务地点",
        "履约地点",
        "实施地点",
    ),
    "project_scope": (
        "project_scope",
        "项目内容",
        "采购内容",
        "服务内容",
        "保险内容",
        "建设内容",
        "招标范围",
        "项目简介",
        "项目概况",
        "采购需求",
    ),
    "service_term": (
        "service_term",
        "合同履行期限",
        "服务期限",
        "保险期限",
        "承保期限",
        "计划工期",
        "工期",
    ),
    "qualification": (
        "qualification",
        "申请人的资格要求",
        "投标人资格要求",
        "供应商资格要求",
        "供应商资格条件",
        "投标人资格",
        "资格条件",
        "报名条件",
    ),
    "key_points": (
        "key_points",
        "商机关键要点",
        "关键要点",
    ),
}

_TITLE_ALIASES = ("project_name", "项目名称", "商机标题", "标题", "title")
_BODY_ALIASES = ("公告正文", "正文", "content", "公告内容")
_SUMMARY_ALIASES = ("内容摘要", "摘要", "summary", "证据摘录", "evidence_excerpt")

_METHOD_PHRASES: tuple[str, ...] = (
    "竞争性磋商",
    "竞争性谈判",
    "框架协议采购",
    "定向询比",
    "公开询比",
    "询比采购",
    "公开招标",
    "邀请招标",
    "单一来源",
    "询价采购",
    "电子卖场",
    "直接采购",
    "比选",
    "询价",
)


def _raw_get(row: Any, key: str) -> Any:
    """兼容普通 Mapping 与 pandas.Series，读取失败时按空值处理。"""

    if isinstance(row, Mapping):
        return row.get(key)
    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            return getter(key, None)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    # pandas.NA 不能直接用于布尔判断，因此只使用安全的字符串哨兵。
    try:
        rendered = str(value).strip()
    except Exception:
        return True
    return rendered.casefold() in _EMPTY_MARKERS


def _clean_text(value: Any, *, limit: int | None = None) -> str:
    """清理 HTML/控制字符，同时保留段落边界供章节提取使用。"""

    if _is_missing(value):
        return ""
    if isinstance(value, datetime):
        text = value.isoformat(sep=" ", timespec="seconds")
    elif isinstance(value, date):
        text = value.isoformat()
    else:
        text = str(value)
    text = html.unescape(text).replace("\ufeff", "").replace("\u200b", "")
    text = _BREAK_TAG_RE.sub("\n", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    if text.casefold() in _EMPTY_MARKERS:
        return ""
    if limit is not None:
        return text[: max(0, int(limit))].rstrip()
    return text


def _compact(value: Any, *, limit: int) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    parts = [part.strip(" ，,。；;：:") for part in text.splitlines()]
    text = "；".join(part for part in parts if part)
    text = re.sub(r"[；;]{2,}", "；", text).strip(" ，,。；;：:")
    return text[:limit].rstrip(" ，,。；;：:")


def _first_explicit(row: Any, aliases: Sequence[str], *, limit: int) -> str:
    for alias in aliases:
        value = _compact(_raw_get(row, alias), limit=limit)
        if value:
            return value
    return ""


def _first_source(row: Any, aliases: Sequence[str]) -> str:
    """返回第一个非空原文字段，不把多个相同摘要重复拼接。"""

    for alias in aliases:
        value = _clean_text(_raw_get(row, alias))
        if value:
            return value
    return ""


def _material(row: Any) -> tuple[str, str, str]:
    title = _first_source(row, _TITLE_ALIASES)
    body = _first_source(row, _BODY_ALIASES)
    summary = _first_source(row, _SUMMARY_ALIASES)
    if body and summary and summary in body:
        summary = ""
    return title, body, summary


def _label_expression(labels: Sequence[str]) -> str:
    return "|".join(re.escape(label) for label in sorted(set(labels), key=len, reverse=True))


def _extract_labeled_section(
    text: str,
    labels: Sequence[str],
    *,
    stops: Sequence[str],
    limit: int,
) -> str:
    """从明确的“标签：值”章节中截取内容。

    冒号允许省略，以兼容“服务期限1096天”之类的公告写法；标签前禁止紧邻
    汉字/字母/数字，避免把短标签误命中在更长词语内部。章节只会在明确的下一个
    标签或中文大标题处停止，不会把资格条款中的 ``1.``、``2.`` 当成新章节。
    """

    source = _clean_text(text)
    if not source:
        return ""
    label_expr = _label_expression(labels)
    label_re = re.compile(
        rf"(?<![\u4e00-\u9fffA-Za-z0-9])(?:{label_expr})"
        r"\s*(?:[（(][^()（）\n]{0,30}[)）])?\s*[：:]?\s*",
        flags=re.IGNORECASE,
    )
    stop_expr = _label_expression(stops) if stops else ""
    stop_patterns: list[re.Pattern[str]] = []
    if stop_expr:
        stop_patterns.append(
            re.compile(
                rf"(?:^|\n|[。；;])\s*(?:[{_HAN_HEADING}]+[、.]\s*)?"
                rf"(?:{stop_expr})\s*[：:]",
                flags=re.IGNORECASE,
            )
        )
        # 从网页复制的正文经常丢失换行和句号，但仍保留“四、采购方式”标题。
        stop_patterns.append(
            re.compile(
                rf"[{_HAN_HEADING}]+[、.]\s*(?:{stop_expr})\s*[：:]",
                flags=re.IGNORECASE,
            )
        )

    for match in label_re.finditer(source):
        # 限制扫描窗口；最后仍会按字段上限硬截断。
        tail = source[match.end() : match.end() + max(2_000, limit * 6)]
        endpoints = [candidate.start() for pattern in stop_patterns for candidate in pattern.finditer(tail)]
        if endpoints:
            tail = tail[: min(endpoints)]
        value = _compact(tail, limit=limit)
        if value:
            return value
    return ""


def _extract_project_number(row: Any, text: str) -> str:
    limit = DETAIL_FIELD_LIMITS["project_number"]
    direct = _first_explicit(row, _FIELD_ALIASES["project_number"], limit=limit)
    if direct:
        return direct
    number_stops = (
        "项目名称",
        "采购项目名称",
        "采购方式",
        "预算金额",
        "招标人",
        "采购人",
        "采购单位",
    )
    # “询价通知书编码”通常只是平台单据编码；正文同时披露询价书/项目编号时，
    # 后者才是业务员需要的项目标识。只有找不到正式编号才回退到通知书编码。
    value = _extract_labeled_section(
        text,
        (
            "项目编号",
            "采购项目编号",
            "采购编号",
            "招标编号",
            "询价书编号",
            "项目代码",
            "标段编号",
        ),
        stops=number_stops,
        limit=limit,
    )
    if not value:
        value = _extract_labeled_section(
            text,
            ("询价通知书编码",),
            stops=number_stops,
            limit=limit,
        )
    if not value:
        return ""
    # 编号不应包含自然语言句子；遇到常见正文标点时保守截断。
    value = re.split(r"[，,。；;\n]", value, maxsplit=1)[0].strip()
    return "" if value.casefold() in _EMPTY_MARKERS else value[:limit]


def _method_from_text(value: str) -> str:
    for phrase in _METHOD_PHRASES:
        if phrase in value:
            return phrase
    return ""


def _extract_procurement_method(row: Any, text: str, title: str) -> str:
    limit = DETAIL_FIELD_LIMITS["procurement_method"]
    direct = _first_explicit(row, _FIELD_ALIASES["procurement_method"], limit=limit)
    if direct:
        return _method_from_text(direct) or direct
    labeled = _extract_labeled_section(
        text,
        ("采购方式", "招标方式", "比选方式", "询价方式"),
        stops=(
            "预算金额",
            "最高限价",
            "报价截止时间",
            "投标截止时间",
            "采购需求",
            "报名须知",
            "报名条件",
        ),
        limit=limit,
    )
    if labeled:
        return _method_from_text(labeled) or labeled
    # 标题本身是公开原文证据，只识别完整的采购方式短语，不根据“公告”猜测。
    return _method_from_text(title)


def _extract_project_location(row: Any, text: str) -> str:
    limit = DETAIL_FIELD_LIMITS["project_location"]
    direct = _first_explicit(row, _FIELD_ALIASES["project_location"], limit=limit)
    if direct:
        return direct
    labeled = _extract_labeled_section(
        text,
        ("项目地点", "建设地点", "服务地点", "履约地点", "实施地点"),
        stops=(
            "项目内容",
            "建设内容",
            "采购内容",
            "招标范围",
            "服务期限",
            "计划工期",
            "采购方式",
        ),
        limit=limit,
    )
    if labeled:
        return labeled

    # 乙方宝标准列已由源文件明确给出，简单并列展示，不从项目名称猜行政区。
    pieces: list[str] = []
    for aliases in (
        ("发布省份", "province"),
        ("发布市级", "city", "地市"),
        ("发布区级", "district", "区县"),
    ):
        piece = _first_explicit(row, aliases, limit=40)
        if piece and piece not in pieces:
            pieces.append(piece)
    return " / ".join(pieces)[:limit]


_SCOPE_STOPS = (
    "采购方式",
    "招标方式",
    "预算金额",
    "最高限价",
    "报价截止时间",
    "投标截止时间",
    "计划工期",
    "合同履行期限",
    "报名须知",
    "报名条件",
    "申请人的资格要求",
    "投标人资格要求",
    "获取采购文件",
    "联系方式",
)


def _extract_project_scope(row: Any, text: str) -> str:
    limit = DETAIL_FIELD_LIMITS["project_scope"]
    direct = _first_explicit(row, _FIELD_ALIASES["project_scope"], limit=limit)
    if direct:
        return direct
    # 先找信息密度更高的内容类标签，把常含报名说明的“项目概况”放在最后。
    # 按标签逐个查找，而不是按其在正文中出现的先后顺序查找。
    for label in (
        "项目简介",
        "采购内容",
        "服务内容",
        "保险内容",
        "建设内容",
        "招标范围",
        "采购需求",
        "项目概况",
    ):
        value = _extract_labeled_section(
            text,
            (label,),
            stops=_SCOPE_STOPS,
            limit=limit,
        )
        if value:
            return value
    return ""


def _extract_service_term(row: Any, text: str) -> str:
    limit = DETAIL_FIELD_LIMITS["service_term"]
    direct = _first_explicit(row, _FIELD_ALIASES["service_term"], limit=limit)
    if direct:
        return direct
    return _extract_labeled_section(
        text,
        ("合同履行期限", "服务期限", "保险期限", "承保期限", "计划工期", "工期"),
        stops=(
            "本项目是否接受联合体",
            "本项目接受联合体",
            "申请人的资格要求",
            "投标人资格要求",
            "供应商资格要求",
            "获取采购文件",
            "采购方式",
            "联系方式",
        ),
        limit=limit,
    )


def _extract_qualification(row: Any, text: str) -> str:
    limit = DETAIL_FIELD_LIMITS["qualification"]
    direct = _first_explicit(row, _FIELD_ALIASES["qualification"], limit=limit)
    if direct:
        return direct
    return _extract_labeled_section(
        text,
        (
            "申请人的资格要求",
            "投标人资格要求",
            "供应商资格要求",
            "供应商资格条件",
            "投标人资格",
            "资格条件",
            "报名条件",
        ),
        stops=(
            "采购单位",
            "招标单位",
            "获取采购文件",
            "招标文件的获取",
            "响应文件提交",
            "提交投标文件截止时间",
            "开启",
            "公告期限",
            "其他补充事宜",
            "联系方式",
        ),
        limit=limit,
    )


def _format_display_value(value: Any, *, amount: bool = False) -> str:
    if _is_missing(value):
        return ""
    if amount and isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return ""
        if numeric.is_integer():
            return f"{int(numeric):,}元"
        return f"{numeric:,.2f}元"
    return _compact(value, limit=160)


def _first_display(row: Any, aliases: Sequence[str], *, amount: bool = False) -> str:
    for alias in aliases:
        value = _format_display_value(_raw_get(row, alias), amount=amount)
        if value:
            return value
    return ""


def _build_key_points(
    row: Any,
    *,
    project_number: str,
    procurement_method: str,
    project_location: str,
    project_scope: str,
    service_term: str,
    qualification: str,
) -> str:
    limit = DETAIL_FIELD_LIMITS["key_points"]
    direct = _first_explicit(row, _FIELD_ALIASES["key_points"], limit=limit)
    if direct:
        return direct

    amount = _first_display(
        row,
        ("招标金额（元）", "招标金额", "预算金额", "amount", "标准金额", "最高限价"),
        amount=True,
    )
    deadline = _first_display(
        row,
        (
            "投标截止原文",
            "投标截止时间",
            "bid_deadline",
            "deadline",
            "报名截止原文",
            "报名截止时间",
            "registration_deadline",
        ),
    )
    tenderer = _first_display(
        row,
        ("招标单位", "采购单位", "招标人", "采购人", "项目业主", "tenderer"),
    )
    agent = _first_display(
        row,
        ("代理单位", "代理机构", "招标代理机构", "采购代理机构", "agent"),
    )

    facts = (
        ("编号", project_number),
        ("方式", procurement_method),
        ("地点", project_location),
        ("金额", amount),
        ("截止", deadline),
        ("采购/招标人", tenderer),
        ("代理", agent),
        ("期限", service_term[:120].rstrip(" ，,。；;：:") if service_term else ""),
        ("资格", qualification[:180].rstrip(" ，,。；;：:") if qualification else ""),
        ("内容", project_scope[:140].rstrip(" ，,。；;：:") if project_scope else ""),
    )
    return _compact(
        "；".join(f"{label}：{value}" for label, value in facts if value),
        limit=limit,
    )


def extract_opportunity_details(row: Any) -> dict[str, str]:
    """从单条商机记录提取可直接展示的关键详情。

    ``row`` 可为普通字典、``Mapping`` 或 pandas ``Series``。返回值始终包含
    :data:`DETAIL_FIELDS` 中的所有字段；没有直接证据的字段返回空字符串。
    """

    title, body, summary = _material(row)
    text = "\n".join(part for part in (body, summary) if part)

    project_number = _extract_project_number(row, text)
    procurement_method = _extract_procurement_method(row, text, title)
    project_location = _extract_project_location(row, text)
    project_scope = _extract_project_scope(row, text)
    service_term = _extract_service_term(row, text)
    qualification = _extract_qualification(row, text)
    key_points = _build_key_points(
        row,
        project_number=project_number,
        procurement_method=procurement_method,
        project_location=project_location,
        project_scope=project_scope,
        service_term=service_term,
        qualification=qualification,
    )

    explicit_status = _first_explicit(
        row,
        ("detail_status", "详情取证状态", "正文取证状态"),
        limit=DETAIL_FIELD_LIMITS["detail_status"],
    )
    if explicit_status:
        detail_status = explicit_status
    elif body:
        detail_status = "正文已提取"
    elif summary:
        detail_status = "摘要已提取"
    elif any((project_number, procurement_method, project_location, key_points)):
        detail_status = "基础字段已提取"
    else:
        detail_status = ""
    detail_source_url = _first_explicit(
        row,
        (
            "detail_source_url",
            "详情来源链接",
            "官网查看地址",
            "官方原文",
            "原文链接",
            "url",
        ),
        limit=DETAIL_FIELD_LIMITS["detail_source_url"],
    )

    result = {
        "project_number": project_number,
        "procurement_method": procurement_method,
        "project_location": project_location,
        "project_scope": project_scope,
        "service_term": service_term,
        "qualification": qualification,
        "key_points": key_points,
        "detail_status": detail_status,
        "detail_source_url": detail_source_url,
    }
    # 防止未来修改无意绕过某个字段的展示上限。
    return {
        field: _compact(result.get(field, ""), limit=DETAIL_FIELD_LIMITS[field])
        for field in DETAIL_FIELDS
    }


def enrich_opportunity_details(frame: Any) -> Any:
    """返回补齐关键详情列的 DataFrame 副本，不修改调用方原表。"""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 项目正式依赖始终包含 pandas
        raise RuntimeError("批量详情提取需要 pandas") from exc
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 pandas.DataFrame")

    result = frame.copy(deep=True)
    if result.empty:
        for field in DETAIL_FIELDS:
            if field not in result.columns:
                result[field] = pd.Series(dtype="object")
        return result

    extracted = [extract_opportunity_details(row) for _, row in result.iterrows()]
    for field in DETAIL_FIELDS:
        result[field] = [record[field] for record in extracted]
    return result


__all__ = [
    "DETAIL_FIELDS",
    "DETAIL_FIELD_LIMITS",
    "extract_opportunity_details",
    "enrich_opportunity_details",
]
