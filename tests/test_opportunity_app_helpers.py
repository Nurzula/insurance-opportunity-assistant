from __future__ import annotations

import pandas as pd

from opportunity_app import (
    _apply_ai_suggestions,
    _formal_output_blockers,
    _merge_editor_changes,
    _records_for_ai,
    _resolve_exact_duplicate_announcements,
    _to_reporting_frame,
)


def test_formal_output_blocks_uncertain_selected_rows_but_allows_unassigned_region() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-1",
                "是否纳入": True,
                "来源类型": "保险",
                "险种分类": "责任险",
                "商机分类": "",
                "需人工复核": False,
                "金额状态": "正常",
                "区域归属": "地区未明确",
                "项目名称": "成都某责任险项目",
            },
            {
                "记录ID": "工程-2",
                "是否纳入": True,
                "来源类型": "工程",
                "险种分类": "工程险",
                "商机分类": "待复核",
                "需人工复核": True,
                "金额状态": "缺失",
                "区域归属": "川内其他地区",
                "项目名称": "标题不明确的项目",
            },
        ]
    )

    blockers = _formal_output_blockers(frame)
    assert blockers["记录ID"].tolist() == ["工程-2"]


def test_reporting_uses_engineering_business_classification() -> None:
    frame = pd.DataFrame(
        [
            {
                "是否纳入": True,
                "来源类型": "工程",
                "险种分类": "工程险",
                "商机分类": "前期线索",
                "项目名称": "某工程招标文件提前公示",
            }
        ]
    )
    result = _to_reporting_frame(frame)
    assert result.loc[0, "category"] == "前期线索"


def test_ai_engineering_categories_are_mapped_back_to_editor_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-1",
                "来源类型": "工程",
                "是否纳入": False,
                "判定状态": "review",
                "需人工复核": True,
                "商机分类": "待复核",
                "复核意见": "",
                "项目名称": "市政道路施工招标公告",
                "内容摘要": "本项目招标范围为市政道路施工。",
                "金额状态": "正常",
            }
        ]
    )
    result = _apply_ai_suggestions(
        frame,
        [
            {
                "record_id": "工程-1",
                "decision": "include",
                "category": "工程直接",
                "confidence": 0.96,
                "reason": "正文证据为“市政道路施工”",
            }
        ],
    )
    assert bool(result.loc[0, "是否纳入"]) is True
    assert result.loc[0, "商机分类"] == "直接施工"
    assert bool(result.loc[0, "需人工复核"]) is False


def test_manual_category_override_clears_internal_confirmation_gate() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-1",
                "来源类型": "保险",
                "是否纳入": False,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "企财险（候选）",
                "商机分类": "",
                "复核意见": "",
            }
        ]
    )
    edited = pd.DataFrame(
        [{"记录ID": "保险-1", "是否纳入": True, "险种分类": "企财险"}]
    )

    result = _merge_editor_changes(frame, edited)
    assert bool(result.loc[0, "是否纳入"]) is True
    assert bool(result.loc[0, "需人工复核"]) is False
    assert result.loc[0, "判定状态"] == "accepted"
    assert "人工确认" in result.loc[0, "复核意见"]
    assert _formal_output_blockers(result).empty


def test_ai_records_keep_public_evidence_and_tolerate_missing_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-public-1",
                "来源类型": "保险",
                "判定状态": "review",
                "金额状态": "正常",
                "项目名称": "雇主责任保险采购公告",
                "招标阶段": "招标公告",
                "标准金额": 85_000,
                "发布市级": "成都",
                "发布区级": "金牛区",
                "公告正文": pd.NA,
                "内容摘要": "采购内容包含雇主责任保险，保险期限一年。",
            }
        ]
    )

    records = _records_for_ai(frame)
    assert len(records) == 1
    assert records[0]["source_type"] == "保险"
    assert "雇主责任保险" in records[0]["excerpt"]


def test_manual_override_can_rescue_an_excluded_record() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-2",
                "来源类型": "保险",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "险种分类": "无关",
                "商机分类": "",
                "AI判定": "exclude",
                "复核意见": "",
            }
        ]
    )
    edited = pd.DataFrame(
        [{"记录ID": "保险-2", "是否纳入": True, "险种分类": "责任险"}]
    )
    result = _merge_editor_changes(frame, edited)
    assert bool(result.loc[0, "是否纳入"]) is True
    assert result.loc[0, "判定状态"] == "accepted"
    assert result.loc[0, "AI判定"] == "人工确认"


def test_formal_output_never_allows_engineering_without_normal_threshold_amount() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-no-money",
                "是否纳入": True,
                "来源类型": "工程",
                "险种分类": "工程险",
                "商机分类": "直接施工",
                "需人工复核": False,
                "金额状态": "缺失",
                "AI判定": "人工确认",
                "项目名称": "道路施工工程",
            }
        ]
    )
    assert _formal_output_blockers(frame, require_ai=True)["记录ID"].tolist() == [
        "工程-no-money"
    ]


def test_ai_high_confidence_without_evidence_anchor_is_not_auto_applied() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-unsupported",
                "来源类型": "工程",
                "项目名称": "道路施工招标公告",
                "内容摘要": "本项目为道路施工。",
                "金额状态": "正常",
                "是否纳入": False,
                "判定状态": "review",
                "需人工复核": True,
                "商机分类": "待复核",
            }
        ]
    )
    result = _apply_ai_suggestions(
        frame,
        [
            {
                "record_id": "工程-unsupported",
                "decision": "include",
                "category": "工程直接",
                "confidence": 0.99,
                "reason": "正文明确采购大型医疗设备",
            }
        ],
    )
    assert bool(result.loc[0, "是否纳入"]) is False
    assert bool(result.loc[0, "需人工复核"]) is True
    assert result.loc[0, "AI理由"] == "正文明确采购大型医疗设备"


def test_public_record_without_full_detail_cannot_enter_formal_output() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-public-failed",
                "是否纳入": True,
                "来源类型": "保险",
                "险种分类": "责任险",
                "商机分类": "",
                "需人工复核": False,
                "金额状态": "正常",
                "AI判定": "include",
                "来源平台": "四川省公共资源交易信息网",
                "正文取证状态": "读取失败",
                "项目名称": "责任保险采购",
            }
        ]
    )
    assert not _formal_output_blockers(frame, require_ai=True).empty


def test_exact_cross_keyword_duplicate_is_merged_after_ai() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-1",
                "公告去重键": "OFFICIAL-1",
                "项目去重键": "PROJECT-1",
                "来源类型": "保险",
                "险种分类": "保证险",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "AI置信度": 0.95,
                "判定理由": "保险规则命中",
            },
            {
                "记录ID": "工程-1",
                "公告去重键": "OFFICIAL-1",
                "项目去重键": "PROJECT-1",
                "来源类型": "工程",
                "险种分类": "工程险",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "AI置信度": 0.98,
                "判定理由": "工程规则命中",
            },
        ]
    )
    result = _resolve_exact_duplicate_announcements(frame)
    assert int(result["是否纳入"].sum()) == 1
    assert bool(result.loc[result["记录ID"].eq("保险-1"), "是否纳入"].iloc[0]) is True
    assert "同一官方公告" in result.loc[result["记录ID"].eq("工程-1"), "判定理由"].iloc[0]
