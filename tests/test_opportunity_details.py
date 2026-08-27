from __future__ import annotations

import pandas as pd

from opportunity_assistant.details import (
    DETAIL_FIELDS,
    DETAIL_FIELD_LIMITS,
    enrich_opportunity_details,
    extract_opportunity_details,
)


def test_extracts_safety_liability_insurance_inquiry_details() -> None:
    title = (
        "中国五冶集团有限公司成都高新未来科技城中村改造项目-"
        "草池社区三期配套道路工程安全生产责任保险招标公告"
    )
    row = {
        "项目名称": title,
        "发布省份": "四川",
        "发布市级": "成都",
        "发布区级": "高新区",
        "招标金额（元）": "",
        "投标截止原文": "2026-08-28 16:00",
        "招标单位": "五矿保险经纪（北京）有限责任公司",
        "官网查看地址": "https://example.test/safety-insurance",
        "公告正文": f"""
            询价通知书编码：20260810980
            询价通知书名称：{title}
            一、询价书编号：WKBX-DXXB-202608024-02
            二、项目名称：{title}
            三、项目简介：{title}，详见询价书。
            四、采购方式：定向询比
            五、报价截止时间：2026-08-28 16:00:00
            六、报名须知：供应商须在电子商务平台进行报名。
            七、报名条件：
            1. 投标人必须是独立的法人单位，具备一般纳税人资格。
            2. 未处于被责令停业、投标资格被取消或财产被接管冻结状态。
            3. 符合中华人民共和国招标投标法等法律法规要求。
            采购单位：五矿保险经纪（北京）有限责任公司
        """,
    }

    details = extract_opportunity_details(row)

    assert details["project_number"] == "WKBX-DXXB-202608024-02"
    assert details["procurement_method"] == "定向询比"
    assert details["project_location"] == "四川 / 成都 / 高新区"
    assert "安全生产责任保险" in details["project_scope"]
    assert "投标人必须是独立的法人单位" in details["qualification"]
    assert "符合中华人民共和国招标投标法" in details["qualification"]
    assert "编号：WKBX-DXXB-202608024-02" in details["key_points"]
    assert "方式：定向询比" in details["key_points"]
    assert "截止：2026-08-28 16:00" in details["key_points"]
    assert "资格：" in details["key_points"]
    assert "投标人必须是独立的法人单位" in details["key_points"]
    assert details["detail_source_url"] == "https://example.test/safety-insurance"


def test_extracts_longquan_education_insurance_procurement_details() -> None:
    row = {
        "项目名称": "成都市龙泉驿区教育局2026年涉教保险采购项目（二次）竞争性磋商公告",
        "发布省份": "四川省",
        "发布市级": "成都市",
        "发布区级": "龙泉驿区",
        "招标金额（元）": 911_435,
        "投标截止时间": "2026-09-07 10:00",
        "招标单位": "成都市龙泉驿区教育局",
        "代理单位": "四川精正建设管理咨询有限公司",
        "公告正文": """
            项目概况：龙泉驿区教育局2026年涉教保险采购项目（二次）的潜在供应商
            应在四川省政府采购一体化平台获取采购文件，并于2026年09月07日10时00分
            前提交响应文件。
            一、项目基本情况
            项目编号：N5101122026000235
            项目名称：龙泉驿区教育局2026年涉教保险采购项目（二次）
            采购方式：竞争性磋商
            预算金额：911,435.00元
            采购需求：详见采购需求附件
            合同履行期限：
            采购包1：365天，服务期限1096天，三年（2026年9月1日—2029年8月31日），
            合同一年一签。
            本项目是否接受联合体参与：采购包1：不接受联合体投标。
            二、申请人的资格要求：
            1. 满足《中华人民共和国政府采购法》第二十二条规定；
            2. 落实政府采购政策需满足的资格要求：本项目不专门面向中小企业采购；
            3. 本项目的特定资格要求：供应商具有开展相关保险业务的许可。
            三、获取采购文件：供应商登录采购平台获取。
        """,
    }

    details = extract_opportunity_details(row)

    assert details["project_number"] == "N5101122026000235"
    assert details["procurement_method"] == "竞争性磋商"
    assert details["project_location"] == "四川省 / 成都市 / 龙泉驿区"
    assert details["project_scope"] == "详见采购需求附件"
    assert "365天" in details["service_term"]
    assert "服务期限1096天" in details["service_term"]
    assert "合同一年一签" in details["service_term"]
    assert "政府采购法" in details["qualification"]
    assert "开展相关保险业务的许可" in details["qualification"]
    assert "金额：911,435元" in details["key_points"]
    assert "截止：2026-09-07 10:00" in details["key_points"]
    assert "采购/招标人：成都市龙泉驿区教育局" in details["key_points"]
    assert "代理：四川精正建设管理咨询有限公司" in details["key_points"]


def test_does_not_invent_details_without_direct_evidence() -> None:
    details = extract_opportunity_details({"项目名称": "某单位设备采购公告"})

    assert set(details) == set(DETAIL_FIELDS)
    assert details["project_number"] == ""
    assert details["procurement_method"] == ""
    assert details["project_location"] == ""
    assert details["project_scope"] == ""
    assert details["service_term"] == ""
    assert details["qualification"] == ""
    assert details["key_points"] == ""
    assert details["detail_status"] == ""


def test_html_is_cleaned_and_every_display_field_obeys_hard_limit() -> None:
    row = {
        "标题": "某责任保险公开招标公告",
        "公告正文": (
            "<p>项目编号：SC-INS-2026-001</p>"
            "<div>采购内容：为员工提供责任保险服务" + "保障说明" * 400 + "</div>"
            "<p>申请人的资格要求：" + "依法合规经营；" * 400 + "</p>"
            "<p>联系方式：采购人办公室</p>"
        ),
    }
    details = extract_opportunity_details(row)

    assert details["project_number"] == "SC-INS-2026-001"
    assert details["procurement_method"] == "公开招标"
    assert "<p>" not in "".join(details.values())
    for field, value in details.items():
        assert len(value) <= DETAIL_FIELD_LIMITS[field]


def test_dataframe_enrichment_returns_copy_and_preserves_row_count() -> None:
    source = pd.DataFrame(
        [
            {
                "项目名称": "某校学生意外保险询价公告",
                "公告正文": "项目编号：XY-1；采购方式：询价；服务期限：一年。",
            },
            {"项目名称": "普通设备公告"},
        ]
    )

    enriched = enrich_opportunity_details(source)

    assert enriched is not source
    assert len(enriched) == len(source)
    assert all(field in enriched.columns for field in DETAIL_FIELDS)
    assert "project_number" not in source.columns
    assert enriched.loc[0, "project_number"] == "XY-1"
    assert enriched.loc[0, "procurement_method"] == "询价"
    assert enriched.loc[0, "service_term"] == "一年"
    assert enriched.loc[1, "project_number"] == ""
