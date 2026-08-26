"""商机推送助手纯业务引擎的离线契约测试。"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from opportunity_assistant import (
    OUTPUT_COLUMNS,
    STANDARD_COLUMNS,
    OpportunityWorkbookError,
    assign_region,
    build_region_summary,
    classify_engineering_dataframe,
    classify_insurance_dataframe,
    clean_amount,
    detect_excel_format,
    make_announcement_key,
    make_project_key,
    parse_yifangbao_excel,
    process_uploaded_workbook,
    split_results,
    summarize_opportunities,
)


INSURANCE_SOURCE = Path(r"D:\downloads\商机信息导出20260826.xls")
ENGINEERING_SOURCE = Path(r"D:\downloads\商机信息导出20260826 (1).xls")


def _source_row(
    title: str,
    *,
    keyword: str = "险",
    amount: object = None,
    province: str = "四川",
    city: str = "成都",
    district: str = "武侯区",
    stage: str = "招标公告",
    url_id: str = "10001",
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in STANDARD_COLUMNS}
    row.update(
        {
            "关键词": keyword,
            "项目名称": title,
            "信息发布时间": date(2026, 8, 25),
            "项目编号": "TEST-001",
            "发布省份": province,
            "发布市级": city,
            "发布区级": district,
            "招标阶段": stage,
            "招标金额（元）": amount,
            "招标单位": "测试招标单位",
            "官网查看地址": f"https://example.test/infoDetail/{url_id}/3484/zhaobiao",
        }
    )
    return row


def _xlsx_bytes(rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "商机信息导出"
    for row in rows:
        sheet.append(row)
    stream = BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def _double_header() -> list[list[object]]:
    first = list(STANDARD_COLUMNS)
    first[11:14] = ["招标单位", "", ""]
    first[14:17] = ["代理单位", "", ""]
    second = list(STANDARD_COLUMNS)
    second[11] = "单位名称"
    return [first, second]


def test_detects_content_not_filename_and_rejects_generic_zip() -> None:
    payload = _xlsx_bytes(_double_header() + [list(_source_row("测试工程").values())])
    assert detect_excel_format(payload) == "xlsx"
    assert detect_excel_format(bytes.fromhex("D0CF11E0A1B11AE1") + b"anything") == "xls"
    with pytest.raises(OpportunityWorkbookError):
        detect_excel_format(b"not an excel workbook")


def test_parses_double_header_xlsx_and_cleans_dates_amounts() -> None:
    values = [
        "工程",
        "道路施工项目招标公告",
        "2026/08/25",
        "P-001",
        "四川",
        "成都",
        "金牛区",
        "招标公告",
        "",
        "2026-09-15 09:30:00",
        "1,234.50万元",
        "建设单位",
        "王老师",
        2800000000,
        "代理单位",
        "李老师",
        "13800000000",
        "https://example.test/infoDetail/123/1/zhaobiao",
    ]
    parsed = parse_yifangbao_excel(_xlsx_bytes(_double_header() + [values]), filename="wrong.xls")

    assert list(parsed.columns) == STANDARD_COLUMNS + ["源文件行号"]
    assert len(parsed) == 1
    assert parsed.at[0, "信息发布时间"] == date(2026, 8, 25)
    assert parsed.at[0, "投标截止时间"] == date(2026, 9, 15)
    assert parsed.at[0, "招标金额（元）"] == pytest.approx(12_345_000)
    assert parsed.at[0, "招标单位联系人电话"] == "2800000000"
    assert parsed.at[0, "源文件行号"] == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,000万元", 10_000_000),
        ("2.5亿元", 250_000_000),
        (85000, 85_000),
        ("--", None),
    ],
)
def test_clean_amount_handles_chinese_units(raw: object, expected: float | None) -> None:
    assert clean_amount(raw) == expected


def test_insurance_rules_keep_eight_categories_and_isolate_false_hits() -> None:
    rows = [
        _source_row("某公司安全生产责任险及第三者责任险采购项目", amount=234_680, url_id="1"),
        _source_row("某单位员工意外及补充医疗保险项目", amount=500, url_id="2"),
        _source_row("工程履约保证金保险机构选聘公告", amount=80_000, url_id="3"),
        _source_row("酒店雇员忠诚险和现金险采购项目", amount=3_250, url_id="4"),
        _source_row("头部带保险孔的圆柱头螺钉招标公告", url_id="5"),
        _source_row("人保财险分公司宣传活动服务采购项目", url_id="6"),
        _source_row("地质灾害风险评估服务项目", url_id="7"),
    ]
    result = classify_insurance_dataframe(pd.DataFrame(rows))
    split = split_results(result)

    assert len(split["accepted"]) == 3
    assert set(split["accepted"]["险种分类"]) == {"责任险", "意外险、健康险", "保证险"}
    assert split["review"]["项目名称"].tolist() == ["酒店雇员忠诚险和现金险采购项目"]
    assert len(split["excluded"]) == 3
    assert result["是否纳入"].dtype == bool
    assert result["需人工复核"].dtype == bool
    assert list(result.columns) == OUTPUT_COLUMNS


def test_public_source_evidence_columns_survive_rule_classification() -> None:
    row = _source_row("雇主责任保险采购公告", amount=85_000, url_id="public-1")
    row.update(
        {
            "数据来源": "四川省公共资源交易信息网",
            "来源平台": "四川省公共资源交易信息网",
            "官方来源标识": "PUBLIC-1",
            "来源分类": "采购公告",
            "公告正文": "本项目采购雇主责任保险，预算金额8.5万元。",
            "内容摘要": "采购雇主责任保险。",
            "金额口径": "预算金额",
            "金额提取依据": "预算金额8.5万元",
        }
    )

    result = classify_insurance_dataframe(pd.DataFrame([row]))
    assert result.loc[0, "官方来源标识"] == "PUBLIC-1"
    assert result.loc[0, "公告正文"].startswith("本项目采购")
    assert result.loc[0, "金额口径"] == "预算金额"
    assert result.loc[0, "金额提取依据"] == "预算金额8.5万元"


@pytest.mark.skipif(not INSURANCE_SOURCE.exists(), reason="本机未提供用户的乙方宝保险源表")
def test_real_insurance_xls_reproduces_department_baseline() -> None:
    result = process_uploaded_workbook(
        INSURANCE_SOURCE.read_bytes(),
        "保险",
        filename=INSURANCE_SOURCE.name,
    )
    accepted = result.loc[result["判定状态"].eq("accepted")]
    review = result.loc[result["判定状态"].eq("review")]

    assert len(result) == 63
    assert len(accepted) == 10
    assert accepted["标准金额"].fillna(0).sum() == pytest.approx(2_207_033)
    assert accepted["项目去重键"].nunique() == 9
    assert review["项目名称"].tolist() == ["成都邛崃智选假日酒店雇员忠诚险和现金险采购项目询价公告"]
    assert review.iloc[0]["险种分类"] == "企财险（候选）"


def test_engineering_rules_cover_threshold_lifecycle_and_bad_amounts() -> None:
    rows = [
        _source_row("市政道路改造工程施工招标公告", keyword="工程", amount=10_000_000, url_id="11"),
        _source_row(
            "新区排水工程招标文件提前公示",
            keyword="工程",
            amount=20_000_000,
            stage="招标预告",
            url_id="12",
        ),
        _source_row("道路施工项目", keyword="工程", amount=9_999_999, url_id="13"),
        _source_row("医院直线加速器设备采购公告", keyword="工程", amount=30_000_000, url_id="14"),
        _source_row("道路施工公告(该信息已更新即将删除)", keyword="工程", amount=30_000_000, url_id="15"),
        _source_row("鸿舰液压分料器制作招标公告", keyword="工程", amount=4.5e17, url_id="16"),
        _source_row("某建设项目招标公告", keyword="工程", amount=None, url_id="17"),
        _source_row("烹坝村1号招标公告", keyword="工程", amount=10_000_000, url_id="18"),
    ]
    result = classify_engineering_dataframe(pd.DataFrame(rows), min_amount=10_000_000)

    assert result["判定状态"].tolist() == [
        "accepted",
        "accepted",
        "excluded",
        "excluded",
        "excluded",
        "review",
        "review",
        "review",
    ]
    assert result["商机分类"].tolist()[:2] == ["直接施工", "前期线索"]
    assert result.loc[5, "金额状态"] == "异常"
    assert result.loc[6, "金额状态"] == "缺失"
    assert result.loc[2, "金额状态"] == "低于门槛"


def test_engineering_installation_and_construction_override_generic_device_words() -> None:
    row = _source_row(
        "泵站设备采购及安装工程施工招标公告",
        keyword="工程",
        amount=20_000_000,
        url_id="installation-1",
    )
    result = classify_engineering_dataframe(pd.DataFrame([row]))
    assert result.loc[0, "判定状态"] == "accepted"
    assert result.loc[0, "商机分类"] == "直接施工"


@pytest.mark.skipif(not ENGINEERING_SOURCE.exists(), reason="本机未提供用户的乙方宝工程源表")
def test_real_engineering_xls_flags_known_bad_source_rows() -> None:
    result = process_uploaded_workbook(
        ENGINEERING_SOURCE.read_bytes(),
        "工程",
        filename=ENGINEERING_SOURCE.name,
    )
    assert len(result) == 55
    bad_amount = result.loc[result["金额状态"].eq("异常")]
    deleted = result.loc[result["项目名称"].str.contains("已更新即将删除", regex=False)]
    assert bad_amount["项目名称"].tolist() == ["鸿舰液压分料器制作招标公告"]
    assert bad_amount.iloc[0]["判定状态"] == "review"
    assert len(deleted) == 1
    assert deleted.iloc[0]["判定状态"] == "excluded"


def test_project_and_announcement_keys_distinguish_lifecycle() -> None:
    base = _source_row(
        "四川发展天盛矿业有限公司2026-2027年度团体意外险服务项目竞价公告",
        city="凉山",
        district="雷波县",
        url_id="625200001",
    )
    amendment = dict(base)
    amendment["项目名称"] = "四川发展天盛矿业有限公司2026-2027年度团体意外险服务项目变更(补遗)公告01"
    amendment["官网查看地址"] = "https://example.test/infoDetail/625200002/1/zhaobiao"

    assert make_project_key(base) == make_project_key(amendment)
    assert make_announcement_key(base) != make_announcement_key(amendment)


@pytest.mark.parametrize(
    ("province", "city", "district", "expected"),
    [
        ("四川", "成都", "成华区", ("成都地区", "无区域类")),
        ("四川", "成都", "锦江区", ("成都地区", "无区域类")),
        ("四川", "成都", "高新区", ("成都地区", "无区域类")),
        ("四川", "成都", "天府新区", ("成都地区", "无区域类")),
        ("四川", "成都", "", ("成都地区", "地区未明确")),
        ("四川", "成都", "金牛区", ("成都地区", "金牛区")),
        ("四川", "宜宾", "叙州区", ("川内其他地区", "宜宾市")),
        ("重庆", "重庆", "渝中区", ("省外", "重庆")),
    ],
)
def test_region_assignment_rules(
    province: str,
    city: str,
    district: str,
    expected: tuple[str, str],
) -> None:
    result = assign_region(province, city, district)
    assert (result["区域大类"], result["区域归属"]) == expected


def test_summary_and_region_table_only_count_accepted_rows() -> None:
    rows = [
        _source_row("团体意外险采购项目", amount=100, district="金牛区", url_id="21"),
        _source_row("公众责任险采购项目", amount=200, district="成华区", url_id="22"),
        _source_row("地质风险评估项目", amount=999, district="金牛区", url_id="23"),
    ]
    result = classify_insurance_dataframe(pd.DataFrame(rows))
    summary = summarize_opportunities(result)
    region = build_region_summary(result)

    assert summary == {
        "raw_count": 3,
        "accepted_count": 2,
        "accepted_amount": 300.0,
        "accepted_unique_project_count": 2,
        "review_count": 0,
        "excluded_count": 1,
        "duplicate_row_count": 0,
    }
    total = region.loc[region["区域归属"].eq("总计")].iloc[0]
    assert total["保险项目数"] == 2
    assert total["保险金额（元）"] == 300
