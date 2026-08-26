"""四川官方公开来源适配器的纯离线测试。"""

from __future__ import annotations

from collections import deque
from datetime import date
import json
from typing import Any

import pandas as pd
import pytest

from opportunity_assistant.core import STANDARD_COLUMNS
from opportunity_assistant.public_sources import (
    HARD_MAX_RECORDS_PER_KEYWORD,
    PUBLIC_COLUMNS,
    SEARCH_ENDPOINT,
    PublicSourceProtocolError,
    PublicSourceRequestError,
    PublicSourceSecurityError,
    SichuanPublicSourceClient,
    clean_html,
    collect_sichuan_public_opportunities,
    detect_sichuan_location,
    enrich_public_dataframe,
    extract_amount,
    extract_engineering_amount,
    extract_official_detail_text,
    fetch_official_detail_text,
    is_active_opportunity,
    map_announcement_stage,
    normalize_public_record,
    validate_official_url,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        url: str = SEARCH_ENDPOINT,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.url = url

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.outcomes:
            raise AssertionError("离线假会话没有剩余响应")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeHtmlResponse:
    def __init__(
        self,
        text: str,
        *,
        status_code: int = 200,
        url: str = "https://ggzyjy.sc.gov.cn/jyxx/test.html",
    ) -> None:
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.url = url
        self.encoding = "utf-8"


class FakeDetailSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeHtmlResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.outcomes:
            raise AssertionError("离线详情会话没有剩余响应")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _record(
    identifier: str,
    title: str,
    *,
    content: str = "",
    link_host: str = "",
    categorynum: str = "002001001",
) -> dict[str, Any]:
    link = f"/jyxx/002001/002001001/20260826/{identifier}.html"
    if link_host:
        link = f"https://{link_host}/record/{identifier}.html"
    return {
        "id": identifier,
        "title": title,
        "content": content,
        "infodate": "2026-08-26 10:20:30",
        "webdate": "2026-08-26 10:00:00",
        "linkurl": link,
        "categorynum": categorynum,
        "zhuanzai": "泸州市公共资源交易中心",
    }


def _page(
    records: list[dict[str, Any]],
    *,
    total: int | None = None,
    result_as_string: bool = False,
) -> FakeResponse:
    result = {
        "totalcount": len(records) if total is None else total,
        "records": records,
        "categorys": [
            {
                "categorynum": "002001001",
                "categoryname": "工程建设-招标公告",
            }
        ],
    }
    return FakeResponse({"result": json.dumps(result, ensure_ascii=False) if result_as_string else result})


def test_html_cleaner_removes_highlights_scripts_and_decodes_entities() -> None:
    source = (
        "<p>采购&nbsp;<em style='color:red'>工程</em></p>"
        "<script>alert('secret')</script><div>第二段<br>下一行</div>"
    )
    cleaned = clean_html(source)
    assert "采购 工程" in cleaned
    assert "第二段" in cleaned and "下一行" in cleaned
    assert "alert" not in cleaned and "script" not in cleaned


def test_detail_extractor_reads_only_news_text_container() -> None:
    html = """
    <html><body><nav>不应进入正文</nav>
      <div id="newsText"><h2>道路改造施工招标公告</h2>
        <div><p>项目预算：1,500万元</p><script>secret()</script></div>
      </div><footer>也不应进入正文</footer></body></html>
    """
    detail = extract_official_detail_text(html)
    assert "道路改造施工招标公告" in detail
    assert "项目预算：1,500万元" in detail
    assert "不应进入正文" not in detail
    assert "也不应进入正文" not in detail
    assert "secret" not in detail


def test_amount_extraction_uses_business_label_priority_not_first_number() -> None:
    text = (
        "项目总投资约5000万元。采购预算：1,234.50万元；"
        "最高投标限价为1200万元，招标控制价1100万元。"
    )
    amount, amount_type, evidence = extract_amount(text)
    assert amount == pytest.approx(12_345_000)
    assert amount_type == "预算金额"
    assert "采购预算" in evidence and "1,234.50万元" in evidence


def test_engineering_scale_prefers_project_investment_over_service_budget() -> None:
    amount, amount_type, evidence = extract_engineering_amount(
        "项目总投资5000万元，本次监理服务采购预算300万元。"
    )
    assert amount == pytest.approx(50_000_000)
    assert amount_type == "投资额"
    assert "项目总投资" in evidence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("最高限价：9800000元", (9_800_000, "最高限价")),
        ("标段投资金额约4982.915308万元", (49_829_153.08, "投资额")),
        ("本项目未公布金额", (None, "")),
    ],
)
def test_amount_extraction_units(text: str, expected: tuple[float | None, str]) -> None:
    amount, amount_type, _ = extract_amount(text)
    if expected[0] is None:
        assert amount is None
    else:
        assert amount == pytest.approx(expected[0])
    assert amount_type == expected[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[四川省﹒泸州市﹒江阳区]老旧街区改造项目", ("泸州", "江阳区")),
        ("成都市金牛区道路建设工程", ("成都", "金牛区")),
        ("凉山彝族自治州昭觉县应急项目", ("凉山", "昭觉县")),
        ("武侯区医疗责任险采购公告", ("成都", "武侯区")),
    ],
)
def test_sichuan_city_and_district_detection(
    text: str, expected: tuple[str, str]
) -> None:
    assert detect_sichuan_location(text) == expected


def test_stage_mapping_and_result_filter_are_conservative() -> None:
    assert map_announcement_stage("某项目采购更正公告") == "信息变更"
    assert map_announcement_stage("某项目第二次采购更正公告") == "信息变更"
    assert map_announcement_stage("某工程招标计划公告") == "招标预告"
    assert map_announcement_stage("某工程补遗澄清公告") == "答疑公告"
    assert map_announcement_stage("某工程施工招标公告") == "招标公告"
    assert is_active_opportunity("某工程施工招标公告") is True
    assert is_active_opportunity("某工程中标候选人公示") is False
    assert is_active_opportunity("某项目合同公告") is False
    assert is_active_opportunity("某项目流标公告") is False
    assert is_active_opportunity("建设工程中标保证保险采购公告") is True


def test_url_whitelist_accepts_only_exact_official_host() -> None:
    assert validate_official_url("/jyxx/test.html") == (
        "https://ggzyjy.sc.gov.cn/jyxx/test.html"
    )
    assert validate_official_url("http://ggzyjy.sc.gov.cn/jyxx/test.html") == (
        "https://ggzyjy.sc.gov.cn/jyxx/test.html"
    )
    with pytest.raises(PublicSourceSecurityError):
        validate_official_url("https://evil.example/jyxx/test.html")
    with pytest.raises(PublicSourceSecurityError):
        validate_official_url("https://ggzyjy.sc.gov.cn.evil.example/test")
    with pytest.raises(PublicSourceSecurityError):
        validate_official_url("//evil.example/test")


def test_normalizes_official_record_to_existing_schema_with_evidence() -> None:
    record = _record(
        "official-001",
        "<em>泸州市江阳区老旧街区改造项目施工招标公告</em>",
        content=(
            "<p>项目编号：LZ-2026-001</p>"
            "<p>采购预算：1,500万元</p>"
            "<p>采购人：泸州某建设有限公司</p>"
            "<p>联系人：王先生 联系电话：0830-1234567</p>"
            "<p>投标截止时间：2026年09月15日09时30分</p>"
        ),
    )
    normalized = normalize_public_record(
        record,
        keyword="工程",
        category_names={"002001001": "工程建设-招标公告"},
    )

    assert all(column in normalized for column in STANDARD_COLUMNS)
    assert all(column in normalized for column in PUBLIC_COLUMNS)
    assert normalized["项目名称"] == "泸州市江阳区老旧街区改造项目施工招标公告"
    assert normalized["项目编号"] == "LZ-2026-001"
    assert normalized["发布省份"] == "四川"
    assert (normalized["发布市级"], normalized["发布区级"]) == ("泸州", "江阳区")
    assert normalized["招标金额（元）"] == pytest.approx(15_000_000)
    assert normalized["金额口径"] == "预算金额"
    assert normalized["招标单位"] == "泸州某建设有限公司"
    assert normalized["招标单位联系人"] == "王先生"
    assert normalized["招标单位联系人电话"] == "0830-1234567"
    assert normalized["投标截止时间"] == date(2026, 9, 15)
    assert normalized["投标截止原文"] == "2026-09-15 09:30"
    assert normalized["官网查看地址"].startswith("https://ggzyjy.sc.gov.cn/jyxx/")
    assert normalized["官方来源标识"] == "official-001"
    assert normalized["来源平台"] == "四川省公共资源交易信息网"
    assert "采购预算" in normalized["公告正文"]


def test_deadline_parser_accepts_common_sentence_with_parenthetical_label() -> None:
    record = _record(
        "deadline-001",
        "道路工程施工招标公告",
        content=(
            "投标文件递交的截止时间（投标截止时间，下同）为"
            "2026年09月15日09时30分。"
        ),
    )
    normalized = normalize_public_record(record, keyword="工程")
    assert normalized["投标截止时间"] == date(2026, 9, 15)
    assert normalized["投标截止原文"] == "2026-09-15 09:30"


def test_normalizer_rejects_off_domain_record_link() -> None:
    with pytest.raises(PublicSourceSecurityError):
        normalize_public_record(
            _record("bad", "工程招标公告", link_host="evil.example"),
            keyword="工程",
        )
    missing_link = _record("missing", "工程招标公告")
    missing_link["linkurl"] = ""
    with pytest.raises(PublicSourceProtocolError, match="缺少官方详情链接"):
        normalize_public_record(missing_link, keyword="工程")


def test_client_retries_timeout_and_sends_public_search_payload() -> None:
    active = _record("retry-ok", "成都市金牛区道路工程施工招标公告")
    session = FakeSession([TimeoutError("synthetic timeout"), _page([active])])
    sleeps: list[float] = []
    client = SichuanPublicSourceClient(
        session=session,
        page_size=10,
        max_retries=2,
        request_delay=0,
        sleep=sleeps.append,
    )
    frame, stats = client.search_keyword(
        "工程",
        "2026-08-25",
        "2026-08-26",
        max_records=10,
    )

    assert len(frame) == 1
    assert len(session.calls) == 2
    assert stats["request_count"] == 2
    assert stats["retry_count"] == 1
    assert sleeps == [0.5]
    call = session.calls[-1]
    assert call["url"] == SEARCH_ENDPOINT
    assert call["json"]["token"] == ""
    assert call["json"]["pn"] == 0
    assert call["json"]["rn"] == 10
    assert call["json"]["wd"] == "%E5%B7%A5%E7%A8%8B"
    assert call["json"]["fields"] == "title"
    assert call["json"]["highlights"] == "title"
    assert call["json"]["sdt"] == "2026-08-25 00:00:00"
    assert call["json"]["edt"] == "2026-08-26 23:59:59"


def test_client_does_not_retry_permanent_http_rejection() -> None:
    session = FakeSession([FakeResponse({}, status_code=403)])
    client = SichuanPublicSourceClient(
        session=session,
        max_retries=3,
        request_delay=0,
        sleep=lambda _: (_ for _ in ()).throw(AssertionError("不应退避重试")),
    )
    with pytest.raises(PublicSourceRequestError, match="HTTP 403"):
        client.search_keyword("工程", "2026-08-26", "2026-08-26")
    assert len(session.calls) == 1


def test_client_paginates_to_cap_and_excludes_result_announcements() -> None:
    page_one = _page(
        [
            _record("a1", "道路工程施工招标公告"),
            _record("r1", "道路工程中标公告"),
        ],
        total=5,
        result_as_string=True,
    )
    page_two = _page([_record("a2", "排水管网工程招标公告")], total=5)
    session = FakeSession([page_one, page_two])
    client = SichuanPublicSourceClient(
        session=session,
        page_size=2,
        max_retries=0,
        request_delay=0,
        sleep=lambda _: None,
    )
    frame, stats = client.search_keyword(
        "工程",
        "2026-08-26",
        "2026-08-26",
        max_records=3,
    )

    assert frame["官方来源标识"].tolist() == ["a1", "a2"]
    assert stats["fetched_count"] == 3
    assert stats["active_count"] == 2
    assert stats["excluded_result_count"] == 1
    assert stats["page_count"] == 2
    assert stats["truncated"] is True
    assert [call["json"]["pn"] for call in session.calls] == [0, 2]
    assert [call["json"]["rn"] for call in session.calls] == [2, 1]


def test_client_enforces_keyword_date_and_hard_record_limits() -> None:
    session = FakeSession([_page([], total=50_000)])
    client = SichuanPublicSourceClient(
        session=session,
        max_retries=0,
        request_delay=0,
        sleep=lambda _: None,
    )
    _, stats = client.search_keyword(
        "险",
        "2026-08-26",
        "2026-08-26",
        max_records=99_999,
    )
    assert stats["effective_limit"] == HARD_MAX_RECORDS_PER_KEYWORD
    with pytest.raises(ValueError):
        client.search_keyword("招聘", "2026-08-26", "2026-08-26")
    with pytest.raises(ValueError):
        client.search_keyword("工程", "2026-08-27", "2026-08-26")


def test_collect_returns_four_contract_keys_and_deduplicates_keyword_overlap() -> None:
    same_insurance = _record(
        "same-001",
        "建设工程安全生产责任险采购公告",
        content="采购预算：200万元",
    )
    result_notice = _record("result-001", "某保险服务采购项目成交公告")
    same_engineering = dict(same_insurance)
    same_engineering["content"] = "项目总投资：5000万元"
    session = FakeSession(
        [
            _page([same_insurance, result_notice], total=2),
            _page([same_engineering], total=1),
        ]
    )
    result = collect_sichuan_public_opportunities(
        "2026-08-26",
        "2026-08-26",
        max_records_per_keyword=10,
        session=session,
        page_size=10,
        max_retries=0,
        request_delay=0,
        max_workers=2,
        sleep=lambda _: None,
    )

    assert set(result) == {"insurance", "engineering", "combined", "stats"}
    assert len(result["insurance"]) == 1
    assert len(result["engineering"]) == 1
    assert len(result["combined"]) == 1
    assert list(result["combined"].columns) == PUBLIC_COLUMNS
    assert result["stats"]["keyword_overlap_count"] == 1
    assert result["stats"]["excluded_result_count"] == 1
    assert result["stats"]["request_count"] == 2
    assert result["stats"]["has_errors"] is False
    assert [call["json"]["wd"] for call in session.calls] == [
        "%E9%99%A9",
        "%E5%B7%A5%E7%A8%8B",
    ]


def test_collect_isolates_one_keyword_failure_without_fabricating_records() -> None:
    session = FakeSession(
        [
            TimeoutError("insurance unavailable"),
            _page([_record("engineering-ok", "市政道路工程施工招标公告")]),
        ]
    )
    result = collect_sichuan_public_opportunities(
        "2026-08-26",
        "2026-08-26",
        session=session,
        max_retries=0,
        request_delay=0,
        max_workers=1,
        sleep=lambda _: None,
    )
    assert result["insurance"].empty
    assert len(result["engineering"]) == 1
    assert len(result["combined"]) == 1
    assert result["stats"]["has_errors"] is True
    assert result["stats"]["keywords"]["险"]["error"]


def test_detail_fetch_and_candidate_enrichment_are_bounded_and_evidence_based() -> None:
    html = """
    <html><body><div id="newsText">
      <p>项目编号：CD-2026-001</p>
      <p>采购预算：1,500万元</p>
      <p>采购人：成都某建设有限公司</p>
      <p>投标截止时间：2026年09月15日09时30分</p>
    </div></body></html>
    """
    session = FakeDetailSession([FakeHtmlResponse(html)])
    detail = fetch_official_detail_text(
        "https://ggzyjy.sc.gov.cn/jyxx/test.html",
        session=session,
        max_retries=0,
    )
    assert "采购预算" in detail

    rows = [
        normalize_public_record(
            _record("candidate", "成都市金牛区道路工程施工招标公告", content="摘要"),
            keyword="工程",
        ),
        normalize_public_record(
            _record("skip", "成都市设备采购公告", content="摘要"),
            keyword="工程",
        ),
    ]
    second_session = FakeDetailSession([FakeHtmlResponse(html)])
    enriched, stats = enrich_public_dataframe(
        pd.DataFrame(rows),
        candidate_mask=[True, False],
        session=second_session,
        max_retries=0,
        request_delay=0,
    )
    assert len(second_session.calls) == 1
    assert enriched.loc[0, "招标金额（元）"] == pytest.approx(15_000_000)
    assert enriched.loc[0, "金额口径"] == "预算金额"
    assert enriched.loc[0, "项目编号"] == "CD-2026-001"
    assert enriched.loc[0, "投标截止时间"] == date(2026, 9, 15)
    assert enriched.loc[0, "投标截止原文"] == "2026-09-15 09:30"
    assert enriched.loc[0, "正文取证状态"] == "完整正文"
    assert enriched.loc[1, "公告正文"] == "摘要"
    assert stats["candidate_count"] == 1
    assert stats["success_row_count"] == 1
    assert stats["failure_url_count"] == 0
