"""本机运行的保险商机智能整理与推送助手。

本入口与 ``app.py`` 的标书核对系统完全隔离。上传文件、处理中间数据和
报告均默认保存在 Streamlit 会话内存中，不写入固定磁盘路径。
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
import hashlib
import io
import os
import re
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from opportunity_assistant.ai_review import review_opportunities
from opportunity_assistant.core import (
    INSURANCE_TYPES,
    assign_regions,
    classify_engineering_dataframe,
    classify_insurance_dataframe,
    parse_yifangbao_excel,
    summarize_opportunities,
)
from opportunity_assistant.reporting import build_report_bundle
from opportunity_assistant.public_sources import (
    PublicSourceError,
    collect_sichuan_public_opportunities,
    enrich_public_dataframe,
)


APP_TITLE = "保险商机智能整理与推送助手"
APP_VERSION = "2.0.0"
DEFAULT_ENGINEERING_MIN_AMOUNT = 10_000_000
DEFAULT_NO_SERVICE_DISTRICTS = "成华区、锦江区、高新区、天府新区"

CORE_EDITABLE_COLUMNS = [
    "是否纳入",
    "险种分类",
    "商机分类",
    "区域归属",
    "复核意见",
    "推送备注",
]

MAX_AI_CANDIDATES = 300


def _setting(name: str, default: str = "") -> str:
    """从本机环境变量或 Streamlit Secrets 安全读取设置。"""

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default) or default).strip()


@st.cache_data(ttl=1_200, max_entries=8, show_spinner=False)
def _cached_public_collect(start_day: date, end_day: date, max_records: int) -> dict[str, Any]:
    """短时缓存同一公开日期查询，避免页面重跑重复访问官网。"""

    return collect_sichuan_public_opportunities(
        start_day,
        end_day,
        max_records_per_keyword=max_records,
    )


@st.cache_data(ttl=1_200, max_entries=16, show_spinner=False)
def _cached_public_enrich(
    frame: pd.DataFrame, candidate_flags: tuple[bool, ...], max_records: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只缓存公开网页正文；会员文件与 AI 判断绝不进入全局缓存。"""

    mask = pd.Series(candidate_flags, index=frame.index)
    return enrich_public_dataframe(
        frame,
        candidate_mask=mask,
        max_records=max_records,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb; }
        [data-testid="stSidebar"] { background: #0b1f3a; }
        [data-testid="stSidebar"] * { color: #eef5ff; }
        .opp-hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(120deg, #0b2f55 0%, #0c6b69 100%);
            box-shadow: 0 10px 30px rgba(11,47,85,.16);
            margin-bottom: 1rem;
        }
        .opp-hero h1 { margin: 0 0 .35rem 0; font-size: 1.85rem; }
        .opp-hero p { margin: 0; opacity: .92; }
        .opp-note {
            padding: .85rem 1rem;
            border-left: 4px solid #13a17d;
            border-radius: 10px;
            background: #eaf8f3;
            color: #15443b;
            margin: .5rem 0 1rem 0;
        }
        .quality-box {
            background: #fff7df;
            border: 1px solid #f1cf6b;
            padding: .8rem 1rem;
            border-radius: 10px;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e0e8f2;
            padding: .8rem 1rem;
            border-radius: 12px;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0b5cab, #0b8f78);
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "opp_results": None,
        "opp_logs": [],
        "opp_bundle": None,
        "opp_report_date": None,
        "opp_ai_reviews": [],
        "opp_ai_summary": {},
        "opp_source_stats": {},
        "opp_input_mode": "",
        "opp_source_hash": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.opp_logs.append(f"[{stamp}] {message}")


def _source_kind(frame: pd.DataFrame) -> str:
    if "关键词" not in frame.columns:
        return ""
    values = [str(value).strip() for value in frame["关键词"].dropna().tolist()]
    if not values:
        return ""
    common = Counter(values).most_common(1)[0][0]
    if common == "险":
        return "保险"
    if common == "工程":
        return "工程"
    return common


def _apply_no_service_rules(frame: pd.DataFrame, districts: Iterable[str]) -> pd.DataFrame:
    """允许管理员在页面调整无服务点名单，不改变核心默认规则。"""

    result = frame.copy()
    normalized = {str(item).strip() for item in districts if str(item).strip()}
    if not normalized or "发布市级" not in result.columns:
        return result

    city = result["发布市级"].fillna("").astype(str).str.strip()
    district = result["发布区级"].fillna("").astype(str).str.strip()
    aliases = {
        "成都高新区": "高新区",
        "高新南区": "高新区",
        "高新西区": "高新区",
        "四川天府新区": "天府新区",
        "成都天府新区": "天府新区",
        "天府新区成都直管区": "天府新区",
    }
    canonical = district.map(lambda value: aliases.get(value, value))
    no_service_mask = city.str.contains("成都", na=False) & canonical.isin(normalized)
    result.loc[no_service_mask, "区域大类"] = "成都地区"
    result.loc[no_service_mask, "区域归属"] = "无区域类"
    return result


def _add_record_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "记录ID" not in result.columns:
        row_values = result.get("源文件行号", pd.Series(range(1, len(result) + 1), index=result.index))
        source_values = result.get("来源类型", pd.Series("商机", index=result.index))
        result["记录ID"] = [f"{source}-{row}" for source, row in zip(source_values, row_values)]
    return result


def _resolve_exact_duplicate_announcements(frame: pd.DataFrame) -> pd.DataFrame:
    """跨“险/工程”命中同一官方公告时只保留一个正式推送记录。"""

    result = frame.copy()
    if "公告去重键" not in result.columns:
        return result
    keys = result["公告去重键"].fillna("").astype(str).str.strip()
    valid_keys = keys.ne("")
    if "项目去重键" in result.columns:
        project_keys = result["项目去重键"].fillna("").astype(str).str.strip()
        result["是否重复"] = project_keys.ne("") & project_keys.duplicated(keep=False)
    for _, indices in result.loc[valid_keys].groupby(keys[valid_keys]).groups.items():
        group_indices = list(indices)
        if len(group_indices) < 2:
            continue
        selected_indices = [
            index for index in group_indices if bool(_selected_mask(result.loc[[index]]).iloc[0])
        ]
        if len(selected_indices) < 2:
            continue

        def rank(index: Any) -> tuple[int, float, int]:
            row = result.loc[index]
            insurance_priority = int(
                row.get("来源类型") == "保险"
                and _valid_insurance_category(row.get("险种分类", ""))
            )
            try:
                confidence = float(row.get("AI置信度", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            return insurance_priority, confidence, -group_indices.index(index)

        keep = max(selected_indices, key=rank)
        for index in selected_indices:
            if index == keep:
                continue
            result.at[index, "是否纳入"] = False
            result.at[index, "判定状态"] = "excluded"
            result.at[index, "需人工复核"] = False
            previous = str(result.at[index, "判定理由"] or "").strip()
            result.at[index, "判定理由"] = (
                f"{previous}；与记录 {result.at[keep, '记录ID']} 为同一官方公告，已自动合并"
            ).strip("；")
    return result


def _most_common_report_date(frame: pd.DataFrame) -> date:
    if frame.empty:
        return date.today()
    for column in ("发布日期", "信息发布时间"):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            modes = values.dt.date.mode()
            if not modes.empty:
                return modes.iloc[0]
    return date.today()


def _money(value: Any) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"¥{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _selected_mask(frame: pd.DataFrame) -> pd.Series:
    if "是否纳入" not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame["是否纳入"]
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).map(lambda value: str(value).strip().lower() in {"true", "1", "是", "纳入", "推送"})


def _valid_insurance_category(value: Any) -> bool:
    parts = [part.strip() for part in str(value or "").replace("，", "、").split("、") if part.strip()]
    return bool(parts) and all(part in INSURANCE_TYPES for part in parts)


def _formal_output_blockers(
    frame: pd.DataFrame, *, require_ai: bool = False
) -> pd.DataFrame:
    """返回不能进入正式推送包的已勾选记录。

    地区缺失或成都无服务点仍可按部门既有口径进入“未分区域项目”；
    只有业务性质、险种或金额本身仍不确定时才阻止生成。
    """

    selected = frame.loc[_selected_mask(frame)].copy()
    if selected.empty:
        return selected
    needs_confirmation = selected.get(
        "需人工复核", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    amount_status_values = selected.get(
        "金额状态", pd.Series("", index=selected.index)
    ).fillna("").astype(str)
    source_type = selected.get(
        "来源类型", pd.Series("", index=selected.index)
    ).fillna("").astype(str)
    abnormal_amount = amount_status_values.eq("异常")
    engineering_bad_amount = source_type.eq("工程") & ~amount_status_values.eq("正常")
    insurance_categories = selected.get(
        "险种分类", pd.Series("", index=selected.index)
    ).fillna("").astype(str)
    insurance_uncertain = source_type.eq("保险") & ~insurance_categories.map(
        _valid_insurance_category
    )
    engineering_uncertain = selected.get(
        "商机分类", pd.Series("", index=selected.index)
    ).fillna("").astype(str).isin({"待复核", "非工程"})
    if require_ai:
        ai_decision = selected.get(
            "AI判定", pd.Series("", index=selected.index)
        ).fillna("").astype(str)
        ai_unverified = ~ai_decision.isin({"include", "人工确认"})
        public_detail_missing = (
            selected.get("来源平台", pd.Series("", index=selected.index))
            .fillna("")
            .astype(str)
            .str.contains("公共资源交易", na=False)
            & ~selected.get("正文取证状态", pd.Series("", index=selected.index))
            .fillna("")
            .astype(str)
            .eq("完整正文")
            & ~ai_decision.eq("人工确认")
        )
    else:
        ai_unverified = pd.Series(False, index=selected.index)
        public_detail_missing = pd.Series(False, index=selected.index)
    return selected.loc[
        needs_confirmation
        | abnormal_amount
        | engineering_bad_amount
        | insurance_uncertain
        | engineering_uncertain
        | ai_unverified
        | public_detail_missing
    ]


def _to_reporting_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """将核心引擎中文列映射为报告模块的稳定英文列。"""

    result = pd.DataFrame(index=frame.index)
    mapping = {
        "selected": "是否纳入",
        "business_type": "来源类型",
        "category": "险种分类",
        "project_name": "项目名称",
        "publish_date": "发布日期",
        "amount": "标准金额",
        "city": "发布市级",
        "district": "发布区级",
        "region_group": "区域大类",
        "service_region": "区域归属",
        "stage": "招标阶段",
        "registration_deadline": "报名截止日期",
        "bid_deadline": "投标截止日期",
        "tenderer": "招标单位",
        "tenderer_contact": "招标单位联系人",
        "tenderer_phone": "招标单位联系人电话",
        "agent": "代理单位",
        "agent_contact": "代理单位联系人",
        "agent_phone": "代理单位联系人电话",
        "url": "官网查看地址",
        "decision_reason": "判定理由",
        "project_key": "项目去重键",
        "record_id": "记录ID",
        "source_row": "源文件行号",
        "note": "推送备注",
        "source_keyword": "关键词",
        "source_platform": "来源平台",
        "official_source_id": "官方来源标识",
        "source_category": "来源分类",
        "amount_basis": "金额口径",
        "amount_evidence": "金额提取依据",
        "evidence_excerpt": "证据摘录",
        "evidence_status": "正文取证状态",
        "ai_decision": "AI判定",
        "ai_confidence": "AI置信度",
        "ai_model": "AI复核模型",
        "ai_reason": "AI理由",
        "announcement_key": "公告去重键",
    }
    for target, source in mapping.items():
        result[target] = frame[source] if source in frame.columns else ""
    for target, raw_column in (
        ("registration_deadline", "报名截止原文"),
        ("bid_deadline", "投标截止原文"),
    ):
        if raw_column in frame.columns:
            raw_values = frame[raw_column].fillna("").astype(str).str.strip()
            result[target] = raw_values.where(raw_values.ne(""), result[target])
    result["selected"] = _selected_mask(frame)
    if "商机分类" in frame.columns:
        # 工程记录的“险种分类”固定是工程险；报告中更有用的是
        # “直接施工/前期线索/非工程/待复核”这一业务判断。
        engineering_mask = frame.get("来源类型", pd.Series("", index=frame.index)).eq("工程")
        result.loc[engineering_mask, "category"] = frame.loc[engineering_mask, "商机分类"]
        empty_category = result["category"].fillna("").astype(str).str.strip().eq("")
        result.loc[empty_category, "category"] = frame.loc[empty_category, "商机分类"]
    quality_parts = []
    for column in ("金额状态", "需人工复核", "复核意见"):
        if column in frame.columns:
            quality_parts.append(frame[column].fillna("").astype(str))
    if quality_parts:
        quality = quality_parts[0]
        for part in quality_parts[1:]:
            quality = quality + "；" + part
        result["quality_issue"] = quality.str.strip("；").str.replace(r"(?:；)+", "；", regex=True)
    else:
        result["quality_issue"] = ""
    result["decision"] = frame.get("判定状态", "")
    return result


def _merge_editor_changes(full_frame: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
    result = full_frame.copy()
    if "记录ID" not in edited.columns or "记录ID" not in result.columns:
        return result
    original_selected = _selected_mask(full_frame)
    original_insurance = full_frame.get(
        "险种分类", pd.Series("", index=full_frame.index)
    ).fillna("").astype(str)
    original_engineering = full_frame.get(
        "商机分类", pd.Series("", index=full_frame.index)
    ).fillna("").astype(str)
    changes = edited.set_index("记录ID")
    ids = result["记录ID"]
    for column in CORE_EDITABLE_COLUMNS:
        if column not in changes.columns or column not in result.columns:
            continue
        mapped = ids.map(changes[column])
        valid = mapped.notna()
        values = mapped.loc[valid]
        if pd.api.types.is_bool_dtype(result[column].dtype):
            result.loc[valid, column] = values.astype(bool).to_numpy(dtype=bool)
        else:
            result.loc[valid, column] = values.to_numpy()

    # 当老师明确勾选并把疑难项改成有效业务分类时，视为已完成内部确认。
    # 若仍是“候选/未确定/非工程”等状态，则保留质量闸门，避免半成品输出。
    selected = _selected_mask(result)
    source = result.get("来源类型", pd.Series("", index=result.index)).fillna("").astype(str)
    insurance_category = result.get("险种分类", pd.Series("", index=result.index)).fillna("").astype(str)
    engineering_category = result.get("商机分类", pd.Series("", index=result.index)).fillna("").astype(str)
    confirmed = selected & (
        (source.eq("保险") & insurance_category.map(_valid_insurance_category))
        | (source.eq("工程") & engineering_category.isin({"直接施工", "前期线索"}))
    )
    manual_change = (
        (selected & ~original_selected.reindex(result.index, fill_value=False))
        | insurance_category.ne(original_insurance.reindex(result.index, fill_value=""))
        | engineering_category.ne(original_engineering.reindex(result.index, fill_value=""))
    )
    if "需人工复核" in result.columns:
        newly_confirmed = confirmed & manual_change
        result.loc[newly_confirmed, "需人工复核"] = False
        if "判定状态" in result.columns:
            result.loc[newly_confirmed, "判定状态"] = "accepted"
        if "复核意见" in result.columns:
            empty_note = result["复核意见"].fillna("").astype(str).str.strip().eq("")
            result.loc[newly_confirmed & empty_note, "复核意见"] = "已由业务人员人工确认纳入"
        if "AI判定" in result.columns:
            result.loc[newly_confirmed, "AI判定"] = "人工确认"
    return result


def _editor(frame: pd.DataFrame, source_type: str, key: str) -> pd.DataFrame:
    subset = frame[frame["来源类型"].eq(source_type)].copy()
    if subset.empty:
        st.info(f"没有{source_type}数据。")
        return frame

    subset["处理状态"] = subset.get("判定状态", "").map(
        {"accepted": "已纳入", "review": "需内部确认", "excluded": "已筛除"}
    ).fillna("需内部确认")
    columns = [
        "记录ID",
        "是否纳入",
        "处理状态",
        "需人工复核",
        "险种分类" if source_type == "保险" else "商机分类",
        "区域归属",
        "发布市级",
        "发布区级",
        "标准金额",
        "招标阶段",
        "项目名称",
        "来源平台",
        "AI置信度",
        "判定理由",
        "证据摘录",
        "官网查看地址",
        "复核意见",
        "推送备注",
    ]
    columns = [column for column in columns if column in subset.columns]
    editor_frame = subset[columns].copy()
    disabled = [
        column
        for column in columns
        if column not in {"是否纳入", "险种分类", "商机分类", "区域归属", "复核意见", "推送备注"}
    ]
    column_config: dict[str, Any] = {
        "记录ID": st.column_config.TextColumn("记录ID", width="small"),
        "是否纳入": st.column_config.CheckboxColumn("推送", help="勾选后写入群文案、图片和Excel"),
        "处理状态": st.column_config.TextColumn("处理状态", width="small"),
        "需人工复核": st.column_config.CheckboxColumn("需内部确认", width="small"),
        "项目名称": st.column_config.TextColumn("项目名称", width="large"),
        "标准金额": st.column_config.NumberColumn("招标金额（元）", format="%,.0f"),
        "判定理由": st.column_config.TextColumn("自动判断依据", width="large"),
        "来源平台": st.column_config.TextColumn("信息来源", width="medium"),
        "AI置信度": st.column_config.NumberColumn("AI置信度（0-1）", format="%.2f", width="small"),
        "证据摘录": st.column_config.TextColumn("正文证据", width="large"),
        "官网查看地址": st.column_config.LinkColumn("官方链接", display_text="打开公告", width="small"),
        "区域归属": st.column_config.TextColumn("区域归属", width="medium"),
        "复核意见": st.column_config.TextColumn("确认意见", width="medium"),
    }
    if "险种分类" in columns:
        column_config["险种分类"] = st.column_config.TextColumn(
            "险种分类",
            help="多个目标险种请用中文顿号连接，例如：意外险、健康险。",
            width="medium",
        )
    if "商机分类" in columns:
        editor_frame["商机分类"] = editor_frame["商机分类"].replace({"待复核": "需确认"})
        column_config["商机分类"] = st.column_config.SelectboxColumn(
            "工程线索",
            options=["直接施工", "前期线索", "非工程", "需确认"],
        )

    edited = st.data_editor(
        editor_frame,
        key=key,
        hide_index=True,
        width="stretch",
        height=min(680, max(240, 44 * (len(editor_frame) + 1))),
        disabled=disabled,
        column_config=column_config,
    )
    if "商机分类" in edited.columns:
        edited["商机分类"] = edited["商机分类"].replace({"需确认": "待复核"})
    return _merge_editor_changes(frame, edited)


def _evidence_excerpt(value: Any, source_type: str, max_length: int = 1_800) -> str:
    """从公开正文中截取与险种、工程性质和金额有关的小段，避免整文上送。"""

    text = " ".join(str(value or "").replace("\xa0", " ").split())
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    if source_type == "保险":
        markers = ("保险", "责任", "意外", "健康", "财产", "货运", "信用", "保证")
    else:
        markers = (
            "预算",
            "最高限价",
            "控制价",
            "投资额",
            "总投资",
            "建设规模",
            "招标范围",
            "施工",
            "工程总承包",
        )
    windows: list[str] = []
    lowered = text.casefold()
    # 每个证据主题先取一个窗口，避免“预算”等高频词占满全部片段，
    # 导致后文真正的施工范围或具体险种永远进不了模型上下文。
    for marker in markers:
        index = lowered.find(marker.casefold())
        if index >= 0:
            left = max(0, index - 180)
            right = min(len(text), index + 420)
            snippet = text[left:right]
            if snippet not in windows:
                windows.append(snippet)
        if len(" … ".join(windows)) >= max_length or len(windows) >= 8:
            break
    if not windows:
        return text[:max_length]
    return " … ".join(windows)[:max_length]


def _first_nonempty_text(*values: Any) -> str:
    """返回第一个非空标量文本，安全处理 pandas 的 NA/NaN。"""

    for value in values:
        if value is None or isinstance(value, (list, tuple, dict, set)):
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            continue
        text = str(value).strip()
        if text and text.casefold() not in {"nan", "nat", "<na>"}:
            return text
    return ""


def _records_for_ai(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """挑选所有可能进入正式推送的记录，并附最小化正文证据。"""

    status = frame.get(
        "判定状态", pd.Series("review", index=frame.index)
    ).fillna("review").astype(str)
    source = frame.get(
        "来源类型", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    amount_state = frame.get(
        "金额状态", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    # 保险误命中很多但每天数量不大，全部交给模型二次语义核验；工程只核验
    # 非低于门槛且并非确定性排除的候选，避免把几百条无关公告逐条送模。
    candidate_mask = source.eq("保险") | (
        source.eq("工程") & ~amount_state.eq("低于门槛") & ~status.eq("excluded")
    )
    candidates = frame[candidate_mask].copy()
    records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        source_type = str(row.get("来源类型", ""))
        raw_evidence = _first_nonempty_text(
            row.get("公告正文", ""),
            row.get("内容摘要", ""),
            row.get("证据摘录", ""),
        )
        records.append(
            {
                "record_id": row.get("记录ID", ""),
                "title": row.get("项目名称", ""),
                "stage": row.get("招标阶段", ""),
                "amount": row.get("标准金额", None),
                "region": f"{row.get('发布市级', '')}/{row.get('发布区级', '')}",
                "source_type": source_type,
                "excerpt": _evidence_excerpt(raw_evidence, source_type),
            }
        )
    return records


def _ai_reason_has_evidence_anchor(reason: str, row: pd.Series) -> bool:
    """要求 AI 理由至少有一个四字符片段能在标题或送审证据中找到。"""

    evidence = (
        _first_nonempty_text(row.get("项目名称", ""))
        + " "
        + _first_nonempty_text(
            row.get("公告正文", ""),
            row.get("内容摘要", ""),
            row.get("证据摘录", ""),
        )
    ).casefold()
    reason_text = str(reason or "").casefold()
    compact_evidence = re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", evidence)
    compact_reason = re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", reason_text)
    if len(compact_evidence) < 4 or len(compact_reason) < 4:
        return False
    generic = {
        "标题明确",
        "正文明确",
        "公告明确",
        "建议纳入",
        "建议排除",
        "属于目标",
        "证据不足",
    }
    for start in range(len(compact_reason) - 3):
        anchor = compact_reason[start : start + 4]
        if anchor in generic or any(anchor in phrase for phrase in generic):
            continue
        if anchor in compact_evidence:
            return True
    return False


def _apply_ai_suggestions(
    frame: pd.DataFrame,
    reviews: list[dict[str, Any]],
    *,
    min_confidence: float = 0.82,
) -> pd.DataFrame:
    """自动应用高置信 AI 结果；低置信项只进入内部确认，不进入正式输出。"""

    result = frame.copy()
    if not reviews:
        return result
    review_by_id = {str(item.get("record_id", "")): item for item in reviews}
    for index, row in result.iterrows():
        suggestion = review_by_id.get(str(row.get("记录ID", "")))
        if not suggestion:
            continue
        decision = str(suggestion.get("decision", "review")).lower()
        category = str(suggestion.get("category", "")).strip()
        confidence = suggestion.get("confidence", "")
        reason = str(suggestion.get("reason", "")).strip()
        try:
            confidence_number = float(confidence)
        except (TypeError, ValueError):
            confidence_number = 0.0
        result.at[index, "AI判定"] = decision
        result.at[index, "AI置信度"] = confidence_number
        result.at[index, "AI复核模型"] = str(suggestion.get("model", ""))
        if reason:
            result.at[index, "AI理由"] = reason

        evidence_supported = _ai_reason_has_evidence_anchor(reason, row)
        public_record = "公共资源交易" in str(row.get("来源平台", "") or "")
        detail_ready = (
            not public_record
            or str(row.get("正文取证状态", "") or "").strip() == "完整正文"
        )
        reliable = confidence_number >= min_confidence and evidence_supported and detail_ready
        amount_state = str(row.get("金额状态", "") or "").strip()
        amount_gate_ok = row.get("来源类型") != "工程" or amount_state in {"", "正常"}
        if decision == "include" and reliable and amount_gate_ok:
            result.at[index, "是否纳入"] = True
            result.at[index, "判定状态"] = "accepted"
            result.at[index, "需人工复核"] = False
        elif decision == "exclude" and reliable:
            result.at[index, "是否纳入"] = False
            result.at[index, "判定状态"] = "excluded"
            result.at[index, "需人工复核"] = False
        else:
            result.at[index, "是否纳入"] = False
            result.at[index, "判定状态"] = "review"
            result.at[index, "需人工复核"] = True
        if category:
            target = "险种分类" if row.get("来源类型") == "保险" else "商机分类"
            if target == "商机分类":
                category = {
                    "工程直接": "直接施工",
                    "前期": "前期线索",
                    "无关": "非工程",
                    "待判断": "待复核",
                }.get(category, category)
            elif category == "待判断":
                category = "未确定"
            if (
                target == "险种分类"
                and decision == "include"
                and _valid_insurance_category(row.get("险种分类", ""))
                and _valid_insurance_category(category)
            ):
                existing_parts = str(row.get("险种分类", "")).split("、")
                category = "、".join(dict.fromkeys(existing_parts + category.split("、")))
            result.at[index, target] = category
        result.at[index, "复核意见"] = (
            f"AI审查（置信度 {confidence_number:.0%}，"
            f"证据锚定{'通过' if evidence_supported else '未通过'}）：{reason}"
        ).strip()
    return result


def _run_mandatory_ai_review(frame: pd.DataFrame) -> pd.DataFrame:
    """对全部候选执行 DeepSeek V4 Flash 批量审查并自动应用高置信结果。"""

    api_key = _setting("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("系统未配置 DeepSeek API Key，无法执行必需的 AI 审查。")

    records = _records_for_ai(frame)
    if not records:
        st.session_state.opp_ai_reviews = []
        st.session_state.opp_ai_summary = {
            "submitted": 0,
            "reviewed": 0,
            "fallback": 0,
        }
        return frame
    if len(records) > MAX_AI_CANDIDATES:
        raise RuntimeError(
            f"规则预筛后仍有 {len(records)} 条 AI 候选，超过单次安全上限 "
            f"{MAX_AI_CANDIDATES} 条。请缩短日期范围，避免长时间等待和异常费用。"
        )

    base_url = _setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = _setting("DEEPSEEK_MODEL", "deepseek-v4-flash")
    all_reviews: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode, source_type in (("insurance", "保险"), ("engineering", "工程")):
        subset = [item for item in records if item.get("source_type") == source_type]
        if not subset:
            continue
        _log(f"正在使用 {model} 批量审查 {len(subset)} 条{source_type}候选。")
        response = review_opportunities(
            subset,
            api_key=api_key,
            review_mode=mode,
            base_url=base_url,
            model=model,
            batch_size=25,
        )
        reviews = list(response.get("reviews") or [])
        for item in reviews:
            item["model"] = model
        all_reviews.extend(reviews)
        summaries.append(response)
        _log(response.get("message") or f"{source_type}AI审查完成。")

    reviewed = sum(item.get("source") == "ai" for item in all_reviews)
    fallback = len(all_reviews) - reviewed
    st.session_state.opp_ai_reviews = all_reviews
    st.session_state.opp_ai_summary = {
        "submitted": len(records),
        "reviewed": reviewed,
        "fallback": fallback,
        "model": model,
        "batches": sum(int(item.get("batch_count", 0)) for item in summaries),
    }
    if not reviewed:
        _log("AI 服务本次未返回有效结构化结果；所有候选均已转为内部人工确认，正式输出仍被锁定。")
    if fallback:
        _log(f"AI有 {fallback} 条未能稳定判定，已自动移入内部确认且不会进入正式推送。")
    return _apply_ai_suggestions(frame, all_reviews)


def _render_sidebar() -> tuple[float, list[str]]:
    st.sidebar.markdown("## 🧭 处理规则")
    st.sidebar.caption(f"商机助手 v{APP_VERSION}｜独立于标书核对系统")
    min_amount = st.sidebar.number_input(
        "工程金额门槛（元）",
        min_value=0,
        value=DEFAULT_ENGINEERING_MIN_AMOUNT,
        step=1_000_000,
        format="%d",
    )
    no_service_text = st.sidebar.text_area(
        "成都无服务点区域",
        value=DEFAULT_NO_SERVICE_DISTRICTS,
        help="支持中文逗号、英文逗号或换行分隔。",
    )
    no_service = [
        part.strip()
        for part in no_service_text.replace("，", ",").replace("\n", ",").split(",")
        if part.strip()
    ]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI 审查")
    api_key = _setting("DEEPSEEK_API_KEY")
    model = _setting("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if api_key:
        st.sidebar.success(f"{model} 已连接，候选商机将自动进行 AI 审查。")
    else:
        st.sidebar.error("未配置 DEEPSEEK_API_KEY，系统不会生成正式商机结果。")
    st.sidebar.caption(
        "模型仅接收标题、阶段、金额、地区和公开正文的相关片段；联系人、电话、邮箱和链接会被移除。"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "会员模式只处理公司合法导出的文件；免费模式只访问无需登录的政府公开来源，不绕过付费墙。"
    )
    return float(min_amount), no_service


def _process_uploads(
    insurance_file: Any,
    engineering_file: Any,
    min_amount: float,
    no_service: list[str],
) -> None:
    insurance_bytes = insurance_file.getvalue()
    engineering_bytes = engineering_file.getvalue()
    config_signature = (
        f"|{min_amount}|{','.join(no_service)}|"
        f"{_setting('DEEPSEEK_MODEL', 'deepseek-v4-flash')}|v2"
    ).encode("utf-8")
    source_hash = hashlib.sha256(
        insurance_bytes + b"|" + engineering_bytes + config_signature
    ).hexdigest()

    if (
        source_hash == st.session_state.get("opp_source_hash")
        and st.session_state.get("opp_results") is not None
    ):
        _log("检测到相同文件和相同规则配置，已复用本次会话中的审查结果。")
        return

    st.session_state.opp_logs = []
    _log("开始校验两份乙方宝导出文件。")
    first = parse_yifangbao_excel(insurance_bytes, filename=insurance_file.name)
    second = parse_yifangbao_excel(engineering_bytes, filename=engineering_file.name)
    first_kind = _source_kind(first)
    second_kind = _source_kind(second)
    _log(f"文件识别：{insurance_file.name}={first_kind or '未知'}，{engineering_file.name}={second_kind or '未知'}。")

    if first_kind == "工程" and second_kind == "保险":
        first, second = second, first
        _log("检测到两个上传位置放反，已自动交换。")
    elif first_kind not in {"保险", ""} or second_kind not in {"工程", ""}:
        raise ValueError("文件关键词与上传位置不匹配，请分别上传关键词“险”和“工程”的乙方宝导出表。")

    _finalize_input_frames(
        first,
        second,
        min_amount=min_amount,
        no_service=no_service,
        source_hash=source_hash,
        input_mode="会员Excel导入",
    )


def _finalize_input_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    min_amount: float,
    no_service: list[str],
    source_hash: str,
    input_mode: str,
    source_stats: dict[str, Any] | None = None,
    report_date_hint: date | None = None,
) -> None:
    """把任意合法来源统一送入规则、区域、AI和报告流水线。"""

    insurance = assign_regions(classify_insurance_dataframe(first))
    engineering = assign_regions(
        classify_engineering_dataframe(second, min_amount=min_amount)
    )
    insurance = _apply_no_service_rules(insurance, no_service)
    engineering = _apply_no_service_rules(engineering, no_service)
    insurance = _add_record_ids(insurance)
    engineering = _add_record_ids(engineering)
    combined = pd.concat([insurance, engineering], ignore_index=True, sort=False)
    if "证据摘录" in combined.columns:
        empty_evidence = combined["证据摘录"].fillna("").astype(str).str.strip().eq("")
        amount_evidence = combined.get(
            "金额提取依据", pd.Series("", index=combined.index)
        ).fillna("").astype(str)
        summaries = combined.get(
            "内容摘要", pd.Series("", index=combined.index)
        ).fillna("").astype(str).str.slice(0, 320)
        combined.loc[empty_evidence, "证据摘录"] = amount_evidence.where(
            amount_evidence.str.strip().ne(""), summaries
        ).loc[empty_evidence]

    _log(
        "保险原始记录 {} 条：自动纳入 {} 条、需人工确认 {} 条。".format(
            len(insurance),
            int(_selected_mask(insurance).sum()),
            int(insurance.get("需人工复核", pd.Series(False, index=insurance.index)).fillna(False).astype(bool).sum()),
        )
    )
    _log(
        "工程原始记录 {} 条：自动纳入 {} 条、需人工确认 {} 条。".format(
            len(engineering),
            int(_selected_mask(engineering).sum()),
            int(engineering.get("需人工复核", pd.Series(False, index=engineering.index)).fillna(False).astype(bool).sum()),
        )
    )
    suspicious = combined.get(
        "金额状态", pd.Series("", index=combined.index)
    ).fillna("").astype(str).ne("正常")
    _log(f"金额或格式质量提示 {int(suspicious.sum())} 条，已放入内部确认视图。")

    combined = _run_mandatory_ai_review(combined)
    combined = _resolve_exact_duplicate_announcements(combined)
    ai_selected = int(_selected_mask(combined).sum())
    ai_pending = int(
        combined.get(
            "需人工复核", pd.Series(False, index=combined.index)
        ).fillna(False).astype(bool).sum()
    )
    _log(f"规则与AI联合审查完成：正式候选 {ai_selected} 条，内部确认 {ai_pending} 条。")

    st.session_state.opp_results = combined
    st.session_state.opp_bundle = None
    st.session_state.opp_report_date = report_date_hint or _most_common_report_date(combined)
    st.session_state.opp_source_hash = source_hash
    st.session_state.opp_source_stats = source_stats or {}
    st.session_state.opp_input_mode = input_mode


def _process_public_sources(
    start_day: date,
    end_day: date,
    max_records: int,
    min_amount: float,
    no_service: list[str],
) -> None:
    """从四川省公共资源交易官方公开平台采集，无需乙方宝会员文件。"""

    st.session_state.opp_logs = []
    _log(
        f"开始采集四川省公共资源交易公开信息：{start_day:%Y-%m-%d} 至 {end_day:%Y-%m-%d}。"
    )
    collected = _cached_public_collect(start_day, end_day, max_records)
    first = collected["insurance"]
    second = collected["engineering"]
    stats = dict(collected.get("stats") or {})
    if stats.get("has_errors"):
        errors = [
            str(item.get("error", "")).strip()
            for item in (stats.get("keywords") or {}).values()
            if str(item.get("error", "")).strip()
        ]
        raise PublicSourceError("；".join(errors) or "官方公开检索未完整返回")
    truncated_keywords = [
        keyword
        for keyword, item in (stats.get("keywords") or {}).items()
        if item.get("truncated")
    ]
    if truncated_keywords:
        raise RuntimeError(
            f"关键词 {'、'.join(truncated_keywords)} 的当日命中超过采集上限，"
            "为避免生成不完整日报，系统已停止。请缩短日期范围或提高采集上限。"
        )
    _log(
        f"官方公开来源采集完成：关键词“险” {len(first)} 条，关键词“工程” {len(second)} 条。"
    )

    # 免费模式采用两级漏斗：标题与摘要先执行确定性预筛，只为可能进入推送的
    # 记录读取官网完整正文。这样既能用正文核验金额与业务性质，又避免逐页抓取
    # 已明确无关或结果类的公告。
    insurance_prefilter = classify_insurance_dataframe(first)
    engineering_prefilter = classify_engineering_dataframe(
        second, min_amount=min_amount
    )
    insurance_detail_mask = pd.Series(True, index=insurance_prefilter.index)
    engineering_detail_mask = (
        ~engineering_prefilter["判定状态"].eq("excluded")
        | engineering_prefilter["金额状态"].eq("正常")
    )
    detail_limit = max(1, min(int(max_records), 2_000))
    _log(
        "正在读取官网候选正文：保险 {} 条、工程 {} 条；已明确无关项不逐页访问。".format(
            int(insurance_detail_mask.sum()), int(engineering_detail_mask.sum())
        )
    )
    first, insurance_detail_stats = _cached_public_enrich(
        first,
        tuple(bool(value) for value in insurance_detail_mask.tolist()),
        detail_limit,
    )
    second, engineering_detail_stats = _cached_public_enrich(
        second,
        tuple(bool(value) for value in engineering_detail_mask.tolist()),
        detail_limit,
    )
    stats["detail"] = {
        "险": insurance_detail_stats,
        "工程": engineering_detail_stats,
    }
    _log(
        "官网正文读取完成：保险 {}/{} 条、工程 {}/{} 条。".format(
            insurance_detail_stats.get("success_row_count", 0),
            insurance_detail_stats.get("candidate_count", 0),
            engineering_detail_stats.get("success_row_count", 0),
            engineering_detail_stats.get("candidate_count", 0),
        )
    )
    if insurance_detail_stats.get("truncated") or engineering_detail_stats.get("truncated"):
        raise RuntimeError("官网正文候选超过安全处理上限，为避免漏判，系统已停止。")
    source_hash = hashlib.sha256(
        f"public|{start_day}|{end_day}|{max_records}|{len(first)}|{len(second)}".encode("utf-8")
    ).hexdigest()
    _finalize_input_frames(
        first,
        second,
        min_amount=min_amount,
        no_service=no_service,
        source_hash=source_hash,
        input_mode="官方公开来源",
        source_stats=stats,
        report_date_hint=end_day,
    )


def _render_metrics(frame: pd.DataFrame) -> None:
    selected = _selected_mask(frame)
    insurance = frame[frame["来源类型"].eq("保险")]
    engineering = frame[frame["来源类型"].eq("工程")]
    insurance_selected = selected.reindex(insurance.index, fill_value=False)
    engineering_selected = selected.reindex(engineering.index, fill_value=False)
    unique_insurance = (
        insurance.loc[insurance_selected, "项目去重键"].replace("", pd.NA).dropna().nunique()
        if "项目去重键" in insurance.columns
        else int(insurance_selected.sum())
    )
    review_count = int(frame.get("需人工复核", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum())
    selected_amount = pd.to_numeric(frame.loc[selected, "标准金额"], errors="coerce").sum()

    cols = st.columns(5)
    cols[0].metric("保险推送公告", int(insurance_selected.sum()), delta=f"独立商机 {unique_insurance}")
    cols[1].metric("工程推送项目", int(engineering_selected.sum()))
    cols[2].metric("需人工确认", review_count)
    cols[3].metric("本次推送金额", _money(selected_amount))
    cols[4].metric("原始记录", len(frame))


def _render_source_coverage() -> None:
    """展示官方源本次检索与正文取证覆盖，不把它误称为乙方宝覆盖率。"""

    stats = dict(st.session_state.opp_source_stats or {})
    keywords = stats.get("keywords") or {}
    if not keywords:
        return
    with st.expander("🌐 官方公开源采集完整性", expanded=False):
        rows: list[dict[str, Any]] = []
        details = stats.get("detail") or {}
        for keyword in ("险", "工程"):
            item = dict(keywords.get(keyword) or {})
            detail = dict(details.get(keyword) or {})
            total = int(item.get("api_total", 0) or 0)
            fetched = int(item.get("fetched_count", 0) or 0)
            rows.append(
                {
                    "关键词": keyword,
                    "官网标题命中": total,
                    "已读取检索记录": fetched,
                    "有效招标线索": int(item.get("active_count", 0) or 0),
                    "结果类已剔除": int(item.get("excluded_result_count", 0) or 0),
                    "候选正文": int(detail.get("candidate_count", 0) or 0),
                    "正文读取成功": int(detail.get("success_row_count", 0) or 0),
                    "检索覆盖": f"{(fetched / total):.1%}" if total else "—",
                    "是否截断": "是" if item.get("truncated") else "否",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "这里衡量的是本次官方检索是否完整，不等于乙方宝商业聚合覆盖率。"
            "只有用同一天乙方宝导出表做对照，才能计算两者的真实交集与覆盖率。"
        )


def _render_quality(frame: pd.DataFrame) -> None:
    issues: list[str] = []
    if "金额状态" in frame.columns:
        counts = frame["金额状态"].fillna("未知").astype(str).value_counts()
        for label, count in counts.items():
            if label != "正常":
                issues.append(f"金额状态“{label}” {count} 条")
    if "发布区级" in frame.columns:
        missing_district = frame["发布区级"].fillna("").astype(str).str.strip().isin({"", "--"}).sum()
        if missing_district:
            issues.append(f"区县缺失 {int(missing_district)} 条")
    if "是否重复" in frame.columns:
        duplicates = frame["是否重复"].fillna(False).astype(bool).sum()
        if duplicates:
            issues.append(f"公告重复/版本关联 {int(duplicates)} 条")
    if issues:
        st.markdown(f"<div class='quality-box'>⚠️ {'；'.join(issues)}。请在生成前完成内部确认。</div>", unsafe_allow_html=True)


def _render_ai_panel(frame: pd.DataFrame) -> pd.DataFrame:
    summary = dict(st.session_state.opp_ai_summary or {})
    if summary:
        cols = st.columns(4)
        cols[0].metric("送审候选", int(summary.get("submitted", 0)))
        cols[1].metric("AI有效返回", int(summary.get("reviewed", 0)))
        cols[2].metric("内部确认", int(summary.get("fallback", 0)))
        cols[3].metric("模型批次", int(summary.get("batches", 0)))
        st.caption(
            f"模型：{summary.get('model', '—')}。AI 已在导入/采集时自动运行，"
            "只有通过质量闸门的结果才能生成正式推送包。"
        )
    reviews = list(st.session_state.opp_ai_reviews or [])
    if reviews:
        display = pd.DataFrame(reviews).rename(
            columns={
                "record_id": "记录ID",
                "category": "AI分类",
                "decision": "AI决定",
                "reason": "证据与理由",
                "confidence": "置信度",
                "source": "返回来源",
                "model": "模型",
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
    else:
        st.info("本次没有需要送交模型的候选记录。")
    return frame


def _render_outputs(frame: pd.DataFrame) -> None:
    st.subheader("③ 生成今日商机包")
    report_date = st.date_input(
        "报告日期",
        value=st.session_state.opp_report_date or date.today(),
        key="opp_date_input",
    )
    if st.button("✨ 一键生成群文案、成都PNG和Excel", type="primary", width="stretch"):
        selected_count = int(_selected_mask(frame).sum())
        blockers = _formal_output_blockers(frame, require_ai=True)
        if selected_count == 0:
            st.warning("当前没有勾选任何推送项目。")
        elif not blockers.empty:
            names = "；".join(blockers["项目名称"].fillna("").astype(str).head(5).tolist())
            more = "……" if len(blockers) > 5 else ""
            st.error(
                f"有 {len(blockers)} 条已勾选项目尚未完成内部确认，系统已阻止生成正式推送包。"
                f"请取消勾选或确认分类/金额后再生成：{names}{more}"
            )
        else:
            with st.spinner("正在内存中生成商机包..."):
                report_frame = _to_reporting_frame(frame)
                bundle = build_report_bundle(
                    report_frame,
                    report_date=report_date,
                    processing_log=st.session_state.opp_logs,
                )
            st.session_state.opp_bundle = bundle
            st.session_state.opp_report_date = report_date
            _log(f"商机包生成完成，共纳入 {selected_count} 条公告。")

    bundle = st.session_state.opp_bundle
    if bundle is None:
        return

    concise_text = getattr(bundle, "concise_text", "")
    full_text = getattr(bundle, "full_text", "")
    excel = getattr(bundle, "excel", None)
    png = getattr(bundle, "png", None)
    date_tag = (st.session_state.opp_report_date or date.today()).strftime("%Y%m%d")

    text_tab, preview_tab = st.tabs(["群发文字", "成都图片预览"])
    with text_tab:
        version = st.radio("文案版本", ["简洁版", "完整版"], horizontal=True, key="message_version")
        chosen_text = concise_text if version == "简洁版" else full_text
        st.text_area("可直接复制到企业微信群", value=chosen_text, height=280)
        st.download_button(
            "📝 下载群发文案.txt",
            data=chosen_text.encode("utf-8-sig"),
            file_name=f"今日商机群发文案-{date_tag}.txt",
            mime="text/plain",
            width="stretch",
        )
    with preview_tab:
        if png is not None:
            png_bytes = png.getvalue() if hasattr(png, "getvalue") else bytes(png)
            st.image(png_bytes, caption="成都地区商机长图", width="stretch")

    download_cols = st.columns(2)
    if excel is not None:
        excel_bytes = excel.getvalue() if hasattr(excel, "getvalue") else bytes(excel)
        download_cols[0].download_button(
            "📥 下载商机推送Excel",
            data=excel_bytes,
            file_name=f"{date_tag}-商机推送.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    if png is not None:
        png_bytes = png.getvalue() if hasattr(png, "getvalue") else bytes(png)
        download_cols[1].download_button(
            "🖼️ 下载成都地区PNG",
            data=png_bytes,
            file_name=f"{date_tag}-成都地区商机.png",
            mime="image/png",
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📡", layout="wide")
    _inject_styles()
    _init_state()
    min_amount, no_service = _render_sidebar()
    api_ready = bool(_setting("DEEPSEEK_API_KEY"))

    st.markdown(
        f"""
        <div class="opp-hero">
          <h1>📡 {APP_TITLE}</h1>
          <p>会员Excel与政府公开来源双模式，一次完成正文取证、AI语义筛选、区域分配、群文案、成都长图与专业Excel。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='opp-note'>🔎 AI为必经审查层：规则先筛掉明显无关项，DeepSeek再批量核验候选。公开模式只读取无需登录的四川省公共资源交易官方信息，不访问乙方宝会员详情。</div>",
        unsafe_allow_html=True,
    )

    st.subheader("① 选择商机来源")
    input_mode = st.radio(
        "来源方式",
        ["会员Excel导入", "官方公开来源（免费）"],
        horizontal=True,
        help="两种方式使用完全相同的规则、AI审查、区域分配和报告格式。",
    )

    if input_mode == "会员Excel导入":
        upload_cols = st.columns(2)
        insurance_file = upload_cols[0].file_uploader(
            "上传关键词“险”的源文件",
            type=["xls", "xlsx"],
            key="insurance_upload",
            help="请直接上传乙方宝下载的原始 .xls，不需要手工修改后缀。",
        )
        engineering_file = upload_cols[1].file_uploader(
            "上传关键词“工程”的源文件",
            type=["xls", "xlsx"],
            key="engineering_upload",
            help="系统会再次执行1000万元门槛与业务性质校验。",
        )
        if st.button(
            "🚀 开始智能整理",
            type="primary",
            width="stretch",
            disabled=not (insurance_file and engineering_file and api_ready),
        ):
            try:
                with st.spinner("正在解析、规则预筛并执行AI审查..."):
                    _process_uploads(
                        insurance_file, engineering_file, min_amount, no_service
                    )
                st.success("整理完成。请确认结果后生成今日商机包。")
            except Exception as exc:
                st.session_state.opp_results = None
                _log(f"处理失败：{type(exc).__name__} - {exc}")
                st.error(f"处理失败：{exc}")
    else:
        st.caption(
            "当前免费源：全国公共资源交易平台（四川省）公开交易信息。"
            "它能提供标题、正文和官方链接，但不保证覆盖乙方宝的全部商业聚合来源。"
        )
        public_cols = st.columns([1, 1, 1])
        previous_day = date.today() - timedelta(days=1)
        start_day = public_cols[0].date_input(
            "开始日期", value=previous_day, key="public_start"
        )
        end_day = public_cols[1].date_input(
            "结束日期", value=previous_day, key="public_end"
        )
        max_records = int(
            public_cols[2].number_input(
                "每个关键词最多采集",
                min_value=50,
                max_value=2000,
                value=2000,
                step=50,
            )
        )
        if st.button(
            "🌐 采集公开商机并智能筛选",
            type="primary",
            width="stretch",
            disabled=not api_ready,
        ):
            try:
                with st.spinner("正在从官方公开平台采集、抽取正文并执行AI审查..."):
                    _process_public_sources(
                        start_day,
                        end_day,
                        max_records,
                        min_amount,
                        no_service,
                    )
                st.success("公开商机采集与智能筛选完成。")
            except (PublicSourceError, ValueError, RuntimeError) as exc:
                st.session_state.opp_results = None
                _log(f"处理失败：{type(exc).__name__} - {exc}")
                st.error(f"处理失败：{exc}")
            except Exception as exc:
                st.session_state.opp_results = None
                _log(f"处理失败：{type(exc).__name__} - {exc}")
                st.error("公开来源暂时不可用，请稍后重试或切换会员Excel模式。")

    if not api_ready:
        st.warning("请先在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY，处理按钮才会启用。")

    if st.session_state.opp_logs:
        with st.expander("📋 实时处理日志", expanded=False):
            st.code("\n".join(st.session_state.opp_logs), language=None)

    frame = st.session_state.opp_results
    if frame is None:
        st.info("请选择一种来源方式并开始处理。")
        return

    st.caption(
        f"本次来源：{st.session_state.opp_input_mode or '—'}。"
        "正式结果均已经过确定性规则和AI双重审查。"
    )

    _render_metrics(frame)
    _render_source_coverage()
    _render_quality(frame)
    st.subheader("② 确认自动筛选与区域分配")
    st.caption("只有“推送”列被勾选的记录会进入最终文案、图片和Excel。所有自动判断都可以人工覆盖。")
    insurance_tab, engineering_tab, excluded_tab, ai_tab = st.tabs(
        ["保险商机", "工程商机", "筛除与质量记录", "AI审查记录"]
    )
    with insurance_tab:
        frame = _editor(frame, "保险", "insurance_editor")
    with engineering_tab:
        frame = _editor(frame, "工程", "engineering_editor")
    with excluded_tab:
        excluded = frame[frame.get("判定状态", "").astype(str).eq("excluded")]
        if excluded.empty:
            st.success("没有筛除记录。")
        else:
            show_columns = [
                column
                for column in [
                    "来源类型",
                    "项目名称",
                    "标准金额",
                    "发布市级",
                    "发布区级",
                    "来源平台",
                    "判定理由",
                    "金额状态",
                ]
                if column in excluded.columns
            ]
            st.dataframe(excluded[show_columns], hide_index=True, width="stretch")
    with ai_tab:
        frame = _render_ai_panel(frame)

    st.session_state.opp_results = frame
    st.markdown("---")
    _render_outputs(frame)
    st.caption("提示：报告是业务线索整理结果，发送前仍应由经办老师完成最终确认。")


if __name__ == "__main__":
    main()
