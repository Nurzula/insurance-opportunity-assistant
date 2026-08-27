"""会员导出记录与官方公开记录的离线高精度关联。

本模块只处理已经在内存中的 :class:`pandas.DataFrame`，不发起网络请求。
关联策略刻意偏保守：宁可不补齐，也不把相似项目的正文错配到会员记录。
只有通过标题以及项目编号、招标单位、发布日期等交叉证据确认的唯一一对一
匹配，才会把官方正文、官方链接和金额证据复制回会员记录。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from html import unescape
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .core import clean_date


__all__ = [
    "enrich_member_dataframe",
    "enrich_member_with_official",
    "match_and_enrich_member_dataframe",
    "normalize_match_title",
]


_TITLE_COLUMNS = ("项目名称", "商机标题", "标题", "project_name")
_PROJECT_NUMBER_COLUMNS = (
    "项目编号",
    "采购项目编号",
    "采购编号",
    "招标编号",
    "标段编号",
    "project_number",
)
_TENDERER_COLUMNS = (
    "招标单位",
    "招标人",
    "采购人",
    "采购单位",
    "项目业主",
    "tenderer",
)
_PUBLISH_DATE_COLUMNS = (
    "信息发布时间",
    "发布日期",
    "发布时间",
    "公告日期",
    "publish_date",
)
_OFFICIAL_URL_COLUMNS = (
    "官网查看地址",
    "官方链接",
    "官方原文",
    "详情来源链接",
    "公告链接",
    "url",
)

# 官方证据字段优先覆盖会员导出中的同名占位值。空的官方字段绝不会清空
# 会员记录已有内容。核心业务字段只在会员记录缺失时补齐，避免改变原始数据。
_EVIDENCE_TRANSFERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("公告正文", ("公告正文", "正文", "完整正文", "detail_text")),
    ("内容摘要", ("内容摘要", "正文摘要", "摘要", "summary")),
    ("证据摘录", ("证据摘录", "正文证据", "evidence_excerpt")),
    ("来源平台", ("来源平台", "信息来源", "source_platform")),
    ("数据来源", ("数据来源", "source_name", "来源平台")),
    ("官方来源标识", ("官方来源标识", "来源ID", "official_source_id")),
    ("来源分类", ("来源分类", "公告分类", "source_category")),
    ("金额口径", ("金额口径", "amount_basis")),
    ("金额提取依据", ("金额提取依据", "金额证据", "amount_evidence")),
    ("金额依据", ("金额依据", "金额提取依据", "金额证据", "amount_evidence")),
    ("正文取证状态", ("正文取证状态", "详情取证状态", "detail_status")),
    ("报名截止原文", ("报名截止原文",)),
    ("投标截止原文", ("投标截止原文",)),
    ("项目地点", ("项目地点", "建设地点", "服务地点", "project_location")),
    ("采购方式", ("采购方式", "招标方式", "procurement_method")),
    ("项目内容", ("项目内容", "采购内容", "建设内容", "招标范围", "project_scope")),
    ("服务期限", ("服务期限", "保险期限", "计划工期", "工期", "service_term")),
    ("资格条件", ("资格条件", "投标人资格", "供应商资格", "qualification")),
    ("商机关键要点", ("商机关键要点", "关键要点", "AI要点", "key_points")),
)

_FILL_ONLY_TRANSFERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("项目编号", _PROJECT_NUMBER_COLUMNS),
    ("招标金额（元）", ("招标金额（元）", "预算金额", "amount")),
    ("招标单位", _TENDERER_COLUMNS),
    ("招标单位联系人", ("招标单位联系人", "招标人联系人", "采购人联系人")),
    (
        "招标单位联系人电话",
        ("招标单位联系人电话", "招标人联系电话", "采购人联系电话"),
    ),
    ("代理单位", ("代理单位", "代理机构")),
    ("代理单位联系人", ("代理单位联系人", "代理机构联系人")),
    ("代理单位联系人电话", ("代理单位联系人电话", "代理机构联系电话")),
    ("信息发布时间", _PUBLISH_DATE_COLUMNS),
    ("报名截止时间", ("报名截止时间", "报名截止日期")),
    ("投标截止时间", ("投标截止时间", "投标截止日期")),
    ("发布省份", ("发布省份", "省份")),
    ("发布市级", ("发布市级", "地市", "城市")),
    ("发布区级", ("发布区级", "区县", "区域")),
)

_GENERIC_TITLE_SUFFIXES = (
    "竞争性磋商采购公告",
    "竞争性磋商公告",
    "竞争性谈判采购公告",
    "竞争性谈判公告",
    "公开招标采购公告",
    "公开招标公告",
    "询价采购公告",
    "询价公告",
    "比选公告",
    "采购公告",
    "招标公告",
)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", "nat", "none", "null", "-", "--"}
    return False


def _text(value: Any) -> str:
    if _is_blank(value):
        return ""
    return str(value).strip()


def _first_value(row: Mapping[str, Any] | pd.Series, columns: Sequence[str]) -> Any:
    for column in columns:
        if column in row:
            value = row.get(column, "")
            if not _is_blank(value):
                return value
    return ""


def _compact_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", unescape(_text(value))).casefold()
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def normalize_match_title(value: Any) -> str:
    """生成用于跨来源关联的保守标题键。

    仅去掉地域方括号前缀、标点和通用的公告类型后缀；``第二次``、标段名、
    更正/结果/终止等生命周期词会被保留，以免把不同公告误认为同一条。
    """

    text = unicodedata.normalize("NFKC", unescape(_text(value))).casefold()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\s*(?:(?:\[[^\]]+\]|【[^】]+】)\s*)+", "", text)
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    for suffix in _GENERIC_TITLE_SUFFIXES:
        suffix_key = _compact_text(suffix)
        if compact.endswith(suffix_key) and len(compact) - len(suffix_key) >= 8:
            compact = compact[: -len(suffix_key)]
            break
    return compact


def _normalize_project_number(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).upper()
    text = re.sub(
        r"^(?:项目编号|采购项目编号|采购编号|招标编号|标段编号)\s*[：:]?\s*",
        "",
        text,
    )
    normalized = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", text)
    # 很短的纯流水号不足以成为跨网站关联证据。
    if len(normalized) < 5 or not re.search(r"\d", normalized):
        return ""
    return normalized


def _normalize_organization(value: Any) -> str:
    return _compact_text(value)


def _extract_date(row: Mapping[str, Any] | pd.Series) -> date | None:
    for column in _PUBLISH_DATE_COLUMNS:
        if column not in row:
            continue
        parsed = clean_date(row.get(column, ""))
        if isinstance(parsed, date):
            return parsed
    return None


def _announcement_lifecycle(title: Any) -> str:
    text = _compact_text(title)
    if re.search(r"终止|流标|废标|撤销", text):
        return "termination"
    if re.search(r"中标|成交|结果|候选人公示", text):
        return "result"
    if re.search(r"更正|变更|补遗|澄清|答疑", text):
        return "change"
    if re.search(r"(?:招标|采购|磋商|谈判|询价|比选|遴选)(?:采购)?公告$", text):
        return "solicitation"
    return "unknown"


@dataclass(frozen=True)
class _Features:
    position: int
    title: str
    title_length: int
    project_number: str
    tenderer: str
    published: date | None
    lifecycle: str


@dataclass(frozen=True)
class _Candidate:
    member_position: int
    official_position: int
    score: int
    method: str
    exact_title: bool
    exact_number: bool


def _features(frame: pd.DataFrame) -> list[_Features]:
    records: list[_Features] = []
    for position in range(len(frame)):
        row = frame.iloc[position]
        raw_title = _first_value(row, _TITLE_COLUMNS)
        records.append(
            _Features(
                position=position,
                title=normalize_match_title(raw_title),
                title_length=len(normalize_match_title(raw_title)),
                project_number=_normalize_project_number(
                    _first_value(row, _PROJECT_NUMBER_COLUMNS)
                ),
                tenderer=_normalize_organization(_first_value(row, _TENDERER_COLUMNS)),
                published=_extract_date(row),
                lifecycle=_announcement_lifecycle(raw_title),
            )
        )
    return records


def _make_candidate(
    member: _Features,
    official: _Features,
    *,
    member_title_counts: Counter[str],
    official_title_counts: Counter[str],
) -> _Candidate | None:
    if not member.title or not official.title:
        return None

    if (
        member.lifecycle != "unknown"
        and official.lifecycle != "unknown"
        and member.lifecycle != official.lifecycle
    ):
        return None

    exact_title = member.title == official.title
    title_ratio = 1.0 if exact_title else SequenceMatcher(
        None, member.title, official.title, autojunk=False
    ).ratio()

    both_numbers = bool(member.project_number and official.project_number)
    exact_number = both_numbers and member.project_number == official.project_number
    if both_numbers and not exact_number:
        return None

    both_tenderers = bool(member.tenderer and official.tenderer)
    exact_tenderer = both_tenderers and member.tenderer == official.tenderer
    # 单位字段如果两侧都有明确值却不一致，就不冒险用相似标题覆盖正文。
    if both_tenderers and not exact_tenderer:
        return None

    date_distance: int | None = None
    if member.published is not None and official.published is not None:
        date_distance = abs((member.published - official.published).days)
        if date_distance > 3:
            return None

    if exact_title:
        score = 72
    elif title_ratio >= 0.985:
        score = 64
    elif title_ratio >= 0.95:
        score = 58
    elif exact_number and title_ratio >= 0.78:
        score = 50
    else:
        return None

    confirmations: list[str] = []
    if exact_number:
        score += 32
        confirmations.append("项目编号")
    if exact_tenderer:
        score += 14
        confirmations.append("招标单位")
    if date_distance == 0:
        score += 10
        confirmations.append("发布日期")
    elif date_distance is not None:
        score += 6
        confirmations.append("发布日期相近")

    unique_long_exact_title = (
        exact_title
        and member.title_length >= 16
        and member_title_counts[member.title] == 1
        and official_title_counts[official.title] == 1
    )

    if exact_number and title_ratio >= 0.78 and score >= 82:
        method = "项目编号+标题"
    elif exact_title and confirmations and score >= 82:
        method = "精确标题+" + "+".join(confirmations)
    elif unique_long_exact_title and not both_numbers and not both_tenderers:
        # 两侧缺少交叉字段时，只接受足够长且各自唯一的完全相同标题。
        score = max(score, 86)
        method = "唯一长标题精确匹配"
    elif title_ratio >= 0.95 and exact_tenderer and date_distance is not None and score >= 82:
        method = "近似标题+招标单位+发布日期"
    else:
        return None

    return _Candidate(
        member_position=member.position,
        official_position=official.position,
        score=score,
        method=method,
        exact_title=exact_title,
        exact_number=exact_number,
    )


def _is_unique_best(candidate: _Candidate, alternatives: Iterable[_Candidate]) -> bool:
    competitors = [item for item in alternatives if item != candidate]
    if not competitors:
        return True
    runner_up = max(competitors, key=lambda item: item.score)
    # 相同分数或差距过小属于歧义；不按原始行序强行挑选。
    return candidate.score - runner_up.score >= 8


def _resolve_matches(
    member_features: Sequence[_Features],
    official_features: Sequence[_Features],
) -> tuple[list[_Candidate], set[int]]:
    member_title_counts = Counter(item.title for item in member_features if item.title)
    official_title_counts = Counter(item.title for item in official_features if item.title)
    candidates: list[_Candidate] = []
    for member in member_features:
        for official in official_features:
            candidate = _make_candidate(
                member,
                official,
                member_title_counts=member_title_counts,
                official_title_counts=official_title_counts,
            )
            if candidate is not None:
                candidates.append(candidate)

    by_member: dict[int, list[_Candidate]] = {}
    by_official: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_member.setdefault(candidate.member_position, []).append(candidate)
        by_official.setdefault(candidate.official_position, []).append(candidate)

    ambiguous_members: set[int] = set()
    accepted: list[_Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.score,
            -int(item.exact_number),
            -int(item.exact_title),
            item.member_position,
            item.official_position,
        ),
    ):
        member_options = by_member[candidate.member_position]
        official_options = by_official[candidate.official_position]
        if not _is_unique_best(candidate, member_options):
            ambiguous_members.add(candidate.member_position)
            continue
        if not _is_unique_best(candidate, official_options):
            ambiguous_members.update(item.member_position for item in official_options)
            continue
        accepted.append(candidate)

    # 理论上唯一最佳已保证一对一；这里再做一次防御性约束。
    unique_matches: list[_Candidate] = []
    used_members: set[int] = set()
    used_official: set[int] = set()
    for candidate in accepted:
        if candidate.member_position in used_members or candidate.official_position in used_official:
            ambiguous_members.add(candidate.member_position)
            continue
        used_members.add(candidate.member_position)
        used_official.add(candidate.official_position)
        unique_matches.append(candidate)
    # “歧义数”只统计最终没有采用匹配的会员行，避免把胜出的唯一最佳行也记入。
    ambiguous_members.difference_update(used_members)
    return unique_matches, ambiguous_members


def _ensure_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = ""


def _copy_value(
    target: pd.DataFrame,
    target_position: int,
    target_column: str,
    source_row: pd.Series,
    aliases: Sequence[str],
    *,
    fill_only: bool,
) -> bool:
    value = _first_value(source_row, aliases)
    if _is_blank(value):
        return False
    _ensure_column(target, target_column)
    column_position = target.columns.get_loc(target_column)
    if fill_only and not _is_blank(target.iat[target_position, column_position]):
        return False
    target.iat[target_position, column_position] = value
    return True


def enrich_member_dataframe(
    member_frame: pd.DataFrame,
    official_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把唯一高置信的官方证据补齐到会员记录。

    两个输入均不会被修改。返回值保持会员记录原顺序和索引。原
    ``官网查看地址``（乙方宝会员页）被保存在新列 ``会员查看地址``；只有匹配
    成功且官方链接非空时，``官网查看地址`` 才会被替换为官方公开链接。
    """

    if not isinstance(member_frame, pd.DataFrame):
        raise TypeError("member_frame 必须是 pandas.DataFrame")
    if not isinstance(official_frame, pd.DataFrame):
        raise TypeError("official_frame 必须是 pandas.DataFrame")

    result = member_frame.copy(deep=True)
    if "会员查看地址" in result.columns:
        existing_member_urls = result["会员查看地址"].copy(deep=True)
    else:
        existing_member_urls = pd.Series("", index=result.index, dtype=object)
    if "官网查看地址" in result.columns:
        original_urls = result["官网查看地址"].copy(deep=True)
    else:
        original_urls = pd.Series("", index=result.index, dtype=object)
        result["官网查看地址"] = ""
    result["会员查看地址"] = [
        existing if not _is_blank(existing) else original
        for existing, original in zip(existing_member_urls, original_urls)
    ]

    member_features = _features(member_frame)
    official_features = _features(official_frame)
    matches, ambiguous_members = _resolve_matches(member_features, official_features)

    methods: Counter[str] = Counter()
    copied_fields = 0
    for match in matches:
        source_row = official_frame.iloc[match.official_position]
        official_url = _first_value(source_row, _OFFICIAL_URL_COLUMNS)
        if not _is_blank(official_url):
            _ensure_column(result, "官网查看地址")
            result.iat[
                match.member_position, result.columns.get_loc("官网查看地址")
            ] = official_url
            copied_fields += 1
        for target_column, aliases in _EVIDENCE_TRANSFERS:
            copied_fields += int(
                _copy_value(
                    result,
                    match.member_position,
                    target_column,
                    source_row,
                    aliases,
                    fill_only=False,
                )
            )
        for target_column, aliases in _FILL_ONLY_TRANSFERS:
            copied_fields += int(
                _copy_value(
                    result,
                    match.member_position,
                    target_column,
                    source_row,
                    aliases,
                    fill_only=True,
                )
            )
        methods[match.method] += 1

    member_rows = len(member_frame)
    matched_rows = len(matches)
    stats: dict[str, Any] = {
        "member_rows": member_rows,
        "official_rows": len(official_frame),
        "matched_rows": matched_rows,
        "unmatched_rows": member_rows - matched_rows,
        "ambiguous_rows": len(ambiguous_members),
        "official_rows_used": len({item.official_position for item in matches}),
        "match_rate": (matched_rows / member_rows) if member_rows else 0.0,
        "copied_fields": copied_fields,
        "match_methods": dict(sorted(methods.items())),
    }
    return result, stats


def enrich_member_with_official(
    member_frame: pd.DataFrame,
    official_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``enrich_member_dataframe`` 的语义化别名。"""

    return enrich_member_dataframe(member_frame, official_frame)


def match_and_enrich_member_dataframe(
    member_frame: pd.DataFrame,
    official_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``enrich_member_dataframe`` 的兼容别名。"""

    return enrich_member_dataframe(member_frame, official_frame)
