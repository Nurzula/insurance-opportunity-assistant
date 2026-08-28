from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from opportunity_assistant.core import classify_insurance_dataframe
from opportunity_assistant.member_enrichment import (
    enrich_member_dataframe,
    normalize_match_title,
)


def _member(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "项目名称": "成都市龙泉驿区教育局2026年涉教保险采购项目（二次）竞争性磋商公告",
        "项目编号": "N5101122026000235",
        "招标单位": "成都市龙泉驿区教育局",
        "信息发布时间": date(2026, 8, 26),
        "招标金额（元）": 911_435.0,
        "官网查看地址": "https://www.yfb.example/info/paid-1",
        "数据来源": "会员Excel导入",
    }
    row.update(overrides)
    return row


def _official(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "项目名称": "【四川省】成都市龙泉驿区教育局2026年涉教保险采购项目（二次）竞争性磋商公告",
        "项目编号": "N5101122026000235",
        "招标单位": "成都市龙泉驿区教育局",
        "信息发布时间": "2026-08-26",
        "招标金额（元）": 911_435.0,
        "官网查看地址": "https://ggzyjy.sc.gov.cn/jyxx/official-1.html",
        "公告正文": "采购预算：911435元。保险服务期限三年。",
        "内容摘要": "涉教保险采购，预算911435元。",
        "来源平台": "四川省公共资源交易信息网",
        "数据来源": "四川省公共资源交易信息网",
        "官方来源标识": "official-1",
        "来源分类": "政府采购",
        "金额口径": "采购预算",
        "金额提取依据": "采购预算：911435元",
        "正文取证状态": "完整正文",
        "投标截止原文": "2026-09-07 10:00",
    }
    row.update(overrides)
    return row


def test_normalize_title_removes_region_and_generic_suffix_but_keeps_round() -> None:
    left = normalize_match_title(
        "[四川省·成都市] 某学校保险采购项目（第二次）公开招标公告"
    )
    right = normalize_match_title("某学校保险采购项目(第二次)")
    first_round = normalize_match_title("某学校保险采购项目（第一次）")

    assert left == right
    assert left != first_round


def test_high_confidence_match_copies_official_evidence_and_preserves_member_url() -> None:
    members = pd.DataFrame([_member()])
    officials = pd.DataFrame([_official()])

    enriched, stats = enrich_member_dataframe(members, officials)

    assert enriched.loc[0, "会员查看地址"] == "https://www.yfb.example/info/paid-1"
    assert enriched.loc[0, "官网查看地址"] == (
        "https://ggzyjy.sc.gov.cn/jyxx/official-1.html"
    )
    assert enriched.loc[0, "公告正文"].startswith("采购预算")
    assert enriched.loc[0, "来源平台"] == "四川省公共资源交易信息网"
    assert enriched.loc[0, "金额提取依据"] == "采购预算：911435元"
    assert enriched.loc[0, "正文取证状态"] == "完整正文"
    # 会员原金额不因补齐动作被覆盖。
    assert enriched.loc[0, "招标金额（元）"] == pytest.approx(911_435)
    assert stats["matched_rows"] == 1
    assert stats["official_rows_used"] == 1
    assert stats["match_rate"] == pytest.approx(1.0)


def test_member_url_survives_the_following_insurance_classification_step() -> None:
    enriched, _ = enrich_member_dataframe(
        pd.DataFrame([_member()]),
        pd.DataFrame([_official()]),
    )

    classified = classify_insurance_dataframe(enriched)

    assert classified.loc[0, "会员查看地址"] == (
        "https://www.yfb.example/info/paid-1"
    )
    assert classified.loc[0, "官网查看地址"] == (
        "https://ggzyjy.sc.gov.cn/jyxx/official-1.html"
    )


def test_inputs_are_not_mutated() -> None:
    members = pd.DataFrame([_member()])
    officials = pd.DataFrame([_official()])
    original_members = members.copy(deep=True)
    original_officials = officials.copy(deep=True)

    enrich_member_dataframe(members, officials)

    pd.testing.assert_frame_equal(members, original_members)
    pd.testing.assert_frame_equal(officials, original_officials)


def test_similar_title_with_different_project_number_is_never_matched() -> None:
    members = pd.DataFrame([_member()])
    officials = pd.DataFrame(
        [
            _official(
                项目名称="成都市龙泉驿区教育局2026年涉教保险采购项目（二次）竞争性磋商公告",
                项目编号="N5101122026000999",
                公告正文="这是另一个编号的项目，绝不能复制。",
            )
        ]
    )

    enriched, stats = enrich_member_dataframe(members, officials)

    assert stats["matched_rows"] == 0
    assert enriched.loc[0, "官网查看地址"] == "https://www.yfb.example/info/paid-1"
    assert "公告正文" not in enriched or enriched.loc[0, "公告正文"] == ""


def test_same_title_different_lifecycle_is_not_matched() -> None:
    members = pd.DataFrame(
        [_member(项目名称="某公司保险服务采购项目公开招标公告", 项目编号="")]
    )
    officials = pd.DataFrame(
        [
            _official(
                项目名称="某公司保险服务采购项目中标结果公告",
                项目编号="",
                招标单位="某公司",
                公告正文="中标结果而不是招标正文。",
            )
        ]
    )

    enriched, stats = enrich_member_dataframe(members, officials)

    assert stats["matched_rows"] == 0
    assert "中标结果" not in str(enriched.get("公告正文", pd.Series([""])).iloc[0])


def test_one_official_row_cannot_be_assigned_to_duplicate_member_rows() -> None:
    duplicated = _member()
    members = pd.DataFrame([duplicated, duplicated.copy()])
    officials = pd.DataFrame([_official()])

    enriched, stats = enrich_member_dataframe(members, officials)

    assert stats["matched_rows"] == 0
    assert stats["ambiguous_rows"] == 2
    assert enriched["官网查看地址"].tolist() == [
        "https://www.yfb.example/info/paid-1",
        "https://www.yfb.example/info/paid-1",
    ]


def test_two_distinct_numbered_projects_match_one_to_one_even_with_similar_titles() -> None:
    members = pd.DataFrame(
        [
            _member(
                项目名称="某集团道路工程一标段招标公告",
                项目编号="CD-ROAD-2026-001",
                招标单位="某集团有限公司",
                官网查看地址="https://yfb.example/1",
            ),
            _member(
                项目名称="某集团道路工程二标段招标公告",
                项目编号="CD-ROAD-2026-002",
                招标单位="某集团有限公司",
                官网查看地址="https://yfb.example/2",
            ),
        ]
    )
    officials = pd.DataFrame(
        [
            _official(
                项目名称="某集团道路工程二标段",
                项目编号="CD-ROAD-2026-002",
                招标单位="某集团有限公司",
                官网查看地址="https://official.example/2",
                公告正文="二标段正文",
            ),
            _official(
                项目名称="某集团道路工程一标段",
                项目编号="CD-ROAD-2026-001",
                招标单位="某集团有限公司",
                官网查看地址="https://official.example/1",
                公告正文="一标段正文",
            ),
        ]
    )

    enriched, stats = enrich_member_dataframe(members, officials)

    assert stats["matched_rows"] == 2
    assert enriched["公告正文"].tolist() == ["一标段正文", "二标段正文"]
    assert enriched["官网查看地址"].tolist() == [
        "https://official.example/1",
        "https://official.example/2",
    ]


def test_unique_long_exact_title_can_match_when_both_sources_lack_other_fields() -> None:
    title = "四川省某大型能源集团2026年度财产保险综合服务采购项目"
    members = pd.DataFrame(
        [{"项目名称": title, "官网查看地址": "https://yfb.example/only"}]
    )
    officials = pd.DataFrame(
        [
            {
                "项目名称": f"【四川省】{title}采购公告",
                "官网查看地址": "https://official.example/only",
                "公告正文": "唯一长标题对应的官方正文",
            }
        ]
    )

    enriched, stats = enrich_member_dataframe(members, officials)

    assert stats["matched_rows"] == 1
    assert enriched.loc[0, "公告正文"] == "唯一长标题对应的官方正文"


def test_missing_member_values_are_filled_but_existing_business_values_are_kept() -> None:
    members = pd.DataFrame(
        [
            _member(
                **{
                    "招标金额（元）": None,
                    "招标单位联系人": "会员联系人",
                }
            )
        ]
    )
    officials = pd.DataFrame(
        [
            _official(
                **{
                    "招标金额（元）": 1_234_567,
                    "招标单位联系人": "官方联系人",
                }
            )
        ]
    )

    enriched, _ = enrich_member_dataframe(members, officials)

    assert enriched.loc[0, "招标金额（元）"] == pytest.approx(1_234_567)
    assert enriched.loc[0, "招标单位联系人"] == "会员联系人"


def test_empty_and_unmatched_frames_return_truthful_statistics() -> None:
    empty, empty_stats = enrich_member_dataframe(
        pd.DataFrame(columns=["项目名称", "官网查看地址"]),
        pd.DataFrame(columns=["项目名称"]),
    )
    assert empty.empty
    assert "会员查看地址" in empty.columns
    assert empty_stats == {
        "member_rows": 0,
        "official_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "ambiguous_rows": 0,
        "official_rows_used": 0,
        "match_rate": 0.0,
        "copied_fields": 0,
        "match_methods": {},
    }

    unmatched, stats = enrich_member_dataframe(
        pd.DataFrame([_member()]),
        pd.DataFrame([_official(项目名称="完全不同的公开项目", 项目编号="OTHER-2026-888")]),
    )
    assert stats["matched_rows"] == 0
    assert stats["unmatched_rows"] == 1
    assert unmatched.loc[0, "会员查看地址"] == "https://www.yfb.example/info/paid-1"


def test_repeated_enrichment_does_not_lose_original_member_url() -> None:
    first, _ = enrich_member_dataframe(
        pd.DataFrame([_member()]), pd.DataFrame([_official()])
    )
    second, stats = enrich_member_dataframe(first, pd.DataFrame([_official()]))

    assert stats["matched_rows"] == 1
    assert second.loc[0, "会员查看地址"] == "https://www.yfb.example/info/paid-1"
    assert second.loc[0, "官网查看地址"].startswith("https://ggzyjy.sc.gov.cn/")


def test_invalid_input_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="member_frame"):
        enrich_member_dataframe([], pd.DataFrame())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="official_frame"):
        enrich_member_dataframe(pd.DataFrame(), [])  # type: ignore[arg-type]
