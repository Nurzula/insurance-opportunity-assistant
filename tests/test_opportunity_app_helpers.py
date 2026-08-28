from __future__ import annotations

import pandas as pd
import pytest

from opportunity_app import (
    _apply_ai_suggestions,
    _apply_inline_confirmations,
    _formal_output_blockers,
    _insurance_member_link_frame,
    _merge_editor_changes,
    _most_common_report_date,
    _preserve_member_engineering_rows,
    _preserve_member_source_urls,
    _records_for_ai,
    _resolve_exact_duplicate_announcements,
    _safe_http_url,
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
                "source": "ai",
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


def test_inline_confirmation_can_fix_candidate_insurance_without_engineering_false_block() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-inline-1",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "企财险（候选）",
                "商机分类": "待复核",
                "标准金额": 165_000,
                "金额状态": "正常",
                "AI判定": "review",
                "AI返回来源": "fallback",
                "复核意见": "",
                "项目名称": "酒店雇员忠诚险和现金险采购项目",
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "记录ID": "保险-inline-1",
                "是否纳入": True,
                "人工确认": True,
                "险种分类": "企财险",
                "商机分类": "待复核",
                "标准金额": 165_000,
            }
        ]
    )

    result = _apply_inline_confirmations(frame, actions)
    assert result.loc[0, "险种分类"] == "企财险"
    assert bool(result.loc[0, "需人工复核"]) is False
    assert result.loc[0, "AI判定"] == "人工确认"
    assert _formal_output_blockers(result, require_ai=True).empty


def test_inline_confirmation_deselects_without_faking_ai_confirmation() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-inline-drop",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "未确定",
                "商机分类": "待复核",
                "标准金额": None,
                "金额状态": "缺失",
                "AI判定": "review",
                "项目名称": "待确认保险项目",
                "复核意见": "",
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "记录ID": "保险-inline-drop",
                "是否纳入": False,
                "人工确认": False,
                "险种分类": "",
                "商机分类": "待复核",
                "标准金额": None,
            }
        ]
    )

    result = _apply_inline_confirmations(frame, actions)
    assert bool(result.loc[0, "是否纳入"]) is False
    assert result.loc[0, "判定状态"] == "excluded"
    assert bool(result.loc[0, "需人工复核"]) is False
    assert result.loc[0, "AI判定"] == "review"
    assert "取消" in result.loc[0, "复核意见"]
    assert _formal_output_blockers(result, require_ai=True).empty


def test_inline_confirmation_recomputes_engineering_amount_and_clears_gate() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-inline-ok",
                "来源类型": "工程",
                "是否纳入": True,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "工程险",
                "商机分类": "待复核",
                "标准金额": None,
                "金额状态": "缺失",
                "AI判定": "review",
                "AI返回来源": "fallback",
                "项目名称": "道路改造工程",
                "复核意见": "",
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "记录ID": "工程-inline-ok",
                "是否纳入": True,
                "人工确认": True,
                "险种分类": "工程险",
                "商机分类": "直接施工",
                "标准金额": 12_000_000,
            }
        ]
    )

    result = _apply_inline_confirmations(frame, actions, min_amount=10_000_000)
    assert result.loc[0, "金额状态"] == "正常"
    assert result.loc[0, "商机分类"] == "直接施工"
    assert bool(result.loc[0, "需人工复核"]) is False
    assert result.loc[0, "AI判定"] == "人工确认"
    assert _formal_output_blockers(
        result, require_ai=True, min_amount=10_000_000
    ).empty


@pytest.mark.parametrize(
    ("amount", "expected_state"),
    [
        (None, "缺失"),
        ("abc", "缺失"),
        (0, "低于门槛"),
        (-1, "异常"),
        (9_999_999, "低于门槛"),
    ],
)
def test_inline_confirmation_never_bypasses_engineering_amount_gate(
    amount: object, expected_state: str
) -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-inline-bad",
                "来源类型": "工程",
                "是否纳入": True,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "工程险",
                "商机分类": "待复核",
                "标准金额": None,
                "金额状态": "缺失",
                "项目名称": "金额待核对工程",
                "复核意见": "",
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "记录ID": "工程-inline-bad",
                "是否纳入": True,
                "人工确认": True,
                "险种分类": "工程险",
                "商机分类": "直接施工",
                "标准金额": amount,
            }
        ]
    )

    result = _apply_inline_confirmations(frame, actions, min_amount=10_000_000)
    assert result.loc[0, "金额状态"] == expected_state
    assert bool(result.loc[0, "需人工复核"]) is True
    assert not _formal_output_blockers(
        result, require_ai=True, min_amount=10_000_000
    ).empty


def test_insurance_member_links_only_include_formally_eligible_insurance() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-link-ok",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "险种分类": "责任险",
                "商机分类": "待复核",
                "标准金额": 85_000,
                "金额状态": "正常",
                "AI判定": "include",
                "AI返回来源": "ai",
                "输入模式": "会员Excel导入",
                "项目名称": "公众责任险采购项目",
                "区域归属": "金牛区",
                "招标阶段": "采购公告",
                "会员查看地址": "https://qiye.example/member/insurance-1",
                "官网查看地址": "https://official.example/insurance-1",
            },
            {
                "记录ID": "保险-link-review",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "review",
                "需人工复核": True,
                "险种分类": "未确定",
                "标准金额": None,
                "金额状态": "缺失",
                "AI判定": "review",
                "AI返回来源": "ai",
                "输入模式": "会员Excel导入",
                "项目名称": "待确认保险项目",
                "会员查看地址": "https://qiye.example/member/review",
            },
            {
                "记录ID": "工程-link-ignore",
                "来源类型": "工程",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "险种分类": "工程险",
                "商机分类": "直接施工",
                "标准金额": 20_000_000,
                "金额状态": "正常",
                "AI判定": "include",
                "AI返回来源": "ai",
                "输入模式": "会员Excel导入",
                "项目名称": "道路施工项目",
                "会员查看地址": "https://qiye.example/member/engineering",
            },
            {
                "记录ID": "保险-link-excluded",
                "来源类型": "保险",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "险种分类": "责任险",
                "标准金额": 100_000,
                "金额状态": "正常",
                "AI判定": "exclude",
                "AI返回来源": "ai",
                "输入模式": "会员Excel导入",
                "项目名称": "已筛除保险项目",
                "会员查看地址": "https://qiye.example/member/excluded",
            },
        ]
    )

    result = _insurance_member_link_frame(frame)

    assert result["项目名称"].tolist() == ["公众责任险采购项目"]
    assert result.loc[0, "会员详情地址"] == (
        "https://qiye.example/member/insurance-1"
    )
    assert result.loc[0, "链接状态"] == "可打开"


def test_insurance_member_links_fall_back_to_exported_url_for_legacy_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-link-legacy",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "险种分类": "意外险",
                "标准金额": 120_000,
                "金额状态": "正常",
                "AI判定": "人工确认",
                "AI返回来源": "fallback",
                "输入模式": "会员Excel导入",
                "项目名称": "团体意外险采购项目",
                "官网查看地址": "https://qiye.example/member/legacy",
            }
        ]
    )

    result = _insurance_member_link_frame(frame)

    assert result.loc[0, "会员详情地址"] == (
        "https://qiye.example/member/legacy"
    )


def test_insurance_member_links_reject_unsafe_source_url() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-link-unsafe",
                "来源类型": "保险",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "险种分类": "企财险",
                "标准金额": 200_000,
                "金额状态": "正常",
                "AI判定": "include",
                "AI返回来源": "ai",
                "输入模式": "会员Excel导入",
                "项目名称": "企业财产保险采购项目",
                "会员查看地址": "javascript:alert(1)",
                "官网查看地址": "https://official.example/not-the-member-url",
            }
        ]
    )

    result = _insurance_member_link_frame(frame)

    assert result.loc[0, "会员详情地址"] == ""
    assert result.loc[0, "链接状态"] == "地址格式不可用"
    assert _safe_http_url("https://user:secret@example.com/private") == ""
    assert _safe_http_url("https://example.com/detail?id=1") == (
        "https://example.com/detail?id=1"
    )


def test_preserve_member_source_urls_runs_before_optional_official_lookup() -> None:
    frame = pd.DataFrame(
        [
            {
                "项目名称": "源表项目一",
                "官网查看地址": "https://qiye.example/member/one",
            },
            {
                "项目名称": "源表项目二",
                "官网查看地址": "https://qiye.example/member/two",
                "会员查看地址": "https://qiye.example/member/already-preserved",
            },
        ]
    )

    result = _preserve_member_source_urls(frame)

    assert result["会员查看地址"].tolist() == [
        "https://qiye.example/member/one",
        "https://qiye.example/member/already-preserved",
    ]
    assert pd.isna(frame.loc[0, "会员查看地址"])


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
                "source": "ai",
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


def test_fallback_include_never_counts_as_completed_ai_review() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "保险-fallback-include",
                "是否纳入": True,
                "来源类型": "保险",
                "险种分类": "责任险",
                "商机分类": "保险商机",
                "需人工复核": False,
                "金额状态": "正常",
                "AI判定": "include",
                "AI返回来源": "fallback",
                "项目名称": "雇主责任险采购",
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


def test_member_engineering_reliable_ai_exclusion_is_applied() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-1",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "某大型医疗设备采购项目",
                "内容摘要": "本次仅采购大型医疗设备，不含安装、改造或施工。",
                "标准金额": 30_000_000,
                "金额状态": "正常",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "商机分类": "非工程",
                "险种分类": "工程险",
            }
        ]
    )

    protected = _preserve_member_engineering_rows(frame)
    assert bool(protected.loc[0, "是否纳入"]) is True
    assert protected.loc[0, "商机分类"] == "工程项目"
    assert len(_records_for_ai(protected)) == 1

    reviewed = _apply_ai_suggestions(
        protected,
        [
            {
                "record_id": "工程-member-1",
                "decision": "exclude",
                "category": "无关",
                "confidence": 0.98,
                "reason": "正文明确“仅采购大型医疗设备”",
                "source": "ai",
            }
        ],
    )
    assert bool(reviewed.loc[0, "是否纳入"]) is False
    assert reviewed.loc[0, "判定状态"] == "excluded"
    assert reviewed.loc[0, "商机分类"] == "非工程"
    assert bool(reviewed.loc[0, "需人工复核"]) is False
    assert _formal_output_blockers(reviewed, require_ai=True).empty


def test_member_engineering_unreliable_ai_exclusion_does_not_silently_delete() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-weak",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "某道路改造工程",
                "内容摘要": "道路及排水工程施工。",
                "金额状态": "正常",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "商机分类": "非工程",
            }
        ]
    )
    protected = _preserve_member_engineering_rows(frame)
    reviewed = _apply_ai_suggestions(
        protected,
        [
            {
                "record_id": "工程-member-weak",
                "decision": "exclude",
                "category": "无关",
                "confidence": 0.99,
                "reason": "没有可核验的正文锚点",
                "source": "ai",
            }
        ],
    )
    assert bool(reviewed.loc[0, "是否纳入"]) is True
    assert reviewed.loc[0, "判定状态"] == "accepted"
    assert reviewed.loc[0, "商机分类"] == "工程项目"
    assert reviewed.loc[0, "AI原始判定"] == "exclude"
    assert reviewed.loc[0, "AI判定"] == "review"
    assert _formal_output_blockers(reviewed, require_ai=True).empty


def test_member_engineering_missing_ai_source_is_not_treated_as_verified() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-unknown-source",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "道路改造工程",
                "内容摘要": "道路改造工程施工。",
                "金额状态": "正常",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "商机分类": "工程项目",
            }
        ]
    )
    reviewed = _apply_ai_suggestions(
        frame,
        [
            {
                "record_id": "工程-member-unknown-source",
                "decision": "include",
                "category": "工程直接",
                "confidence": 0.99,
                "reason": "正文证据为“道路改造工程施工”",
            }
        ],
    )
    assert bool(reviewed.loc[0, "需人工复核"]) is True
    assert reviewed.loc[0, "AI返回来源"] == ""
    assert not _formal_output_blockers(reviewed, require_ai=True).empty


def test_member_engineering_conflicting_exclude_category_is_not_auto_deleted() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-conflict",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "道路改造工程",
                "内容摘要": "道路改造工程施工。",
                "金额状态": "正常",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "商机分类": "工程项目",
            }
        ]
    )
    reviewed = _apply_ai_suggestions(
        frame,
        [
            {
                "record_id": "工程-member-conflict",
                "decision": "exclude",
                "category": "前期",
                "confidence": 0.99,
                "reason": "正文证据为“道路改造工程施工”",
                "source": "ai",
            }
        ],
    )
    assert bool(reviewed.loc[0, "是否纳入"]) is True
    assert reviewed.loc[0, "AI判定"] == "review"
    assert reviewed.loc[0, "商机分类"] == "工程项目"
    assert _formal_output_blockers(reviewed, require_ai=True).empty


def test_member_engineering_fallback_is_blocked_from_formal_output() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-fallback",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "市政道路工程",
                "内容摘要": "市政道路工程施工。",
                "金额状态": "正常",
                "是否纳入": True,
                "判定状态": "accepted",
                "需人工复核": False,
                "商机分类": "工程项目",
            }
        ]
    )
    reviewed = _apply_ai_suggestions(
        frame,
        [
            {
                "record_id": "工程-member-fallback",
                "decision": "review",
                "category": "待判断",
                "confidence": 0,
                "reason": "模型暂不可用",
                "source": "fallback",
            }
        ],
    )
    assert bool(reviewed.loc[0, "是否纳入"]) is True
    assert bool(reviewed.loc[0, "需人工复核"]) is True
    assert reviewed.loc[0, "AI返回来源"] == "fallback"
    assert not _formal_output_blockers(reviewed, require_ai=True).empty


def test_obsolete_member_engineering_is_never_preserved_or_sent_to_ai() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-member-obsolete",
                "输入模式": "会员Excel导入",
                "数据来源": "会员Excel导入",
                "来源类型": "工程",
                "项目名称": "某工程（该信息已更新即将删除）",
                "金额状态": "正常",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "商机分类": "非工程",
                "判定理由": "乙方宝标记为已更新即将删除",
            }
        ]
    )
    result = _preserve_member_engineering_rows(frame)
    assert bool(result.loc[0, "是否纳入"]) is False
    assert _records_for_ai(result) == []

    manually_selected = result.copy()
    manually_selected.loc[0, "是否纳入"] = True
    manually_selected.loc[0, "AI判定"] = "人工确认"
    assert not _formal_output_blockers(manually_selected, require_ai=True).empty


def test_report_date_defaults_to_source_publication_date() -> None:
    frame = pd.DataFrame(
        [
            {"发布日期": "2026-08-26"},
            {"发布日期": "2026-08-26"},
            {"发布日期": "2026-08-25"},
        ]
    )
    assert _most_common_report_date(frame).isoformat() == "2026-08-26"


def test_public_engineering_is_not_covered_by_member_preservation_rule() -> None:
    frame = pd.DataFrame(
        [
            {
                "记录ID": "工程-public-1",
                "输入模式": "官方公开来源",
                "数据来源": "四川省公共资源交易信息网",
                "来源平台": "四川省公共资源交易信息网",
                "来源类型": "工程",
                "项目名称": "大型设备采购项目",
                "内容摘要": "本次仅采购设备。",
                "标准金额": 30_000_000,
                "金额状态": "正常",
                "是否纳入": False,
                "判定状态": "excluded",
                "需人工复核": False,
                "商机分类": "非工程",
                "险种分类": "工程险",
            }
        ]
    )

    result = _preserve_member_engineering_rows(frame)
    assert bool(result.loc[0, "是否纳入"]) is False
    assert _records_for_ai(result) == []
