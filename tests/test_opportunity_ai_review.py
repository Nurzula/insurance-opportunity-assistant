"""商机 AI 复核模块的纯离线契约测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from opportunity_assistant.ai_review import (
    AIReviewProtocolError,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MAX_BATCH_SIZE,
    MAX_EXCERPT_LENGTH,
    build_review_request,
    minimize_records,
    parse_json_object,
    review_opportunities,
    validate_review_object,
)


def _response(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.responder(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.completions = _FakeCompletions(responder)
        self.chat = SimpleNamespace(completions=self.completions)


def _echo_valid_response(kwargs: dict[str, Any]) -> Any:
    sent = json.loads(kwargs["messages"][1]["content"])
    reviews = []
    for item in sent["records"]:
        reviews.append(
            {
                "record_id": item["record_id"],
                "category": "工程直接",
                "decision": "include",
                "reason": "标题明确为施工项目",
                "confidence": 0.96,
            }
        )
    return _response(json.dumps({"reviews": reviews}, ensure_ascii=False))


def test_minimize_and_request_never_send_contact_phone_or_link() -> None:
    records = [
        {
            "记录ID": "A-1",
            "项目名称": "某工程 13812345678 https://secret.example/detail",
            "公告类型": "施工招标",
            "招标金额": 20_000_000,
            "地区": "四川成都",
            "联系人": "不应发送的联系人",
            "联系电话": "028-12345678",
            "url": "https://secret.example/raw",
            "extra": {"token": "not-for-model"},
        }
    ]

    minimized = minimize_records(records)
    assert set(minimized[0]) == {"record_id", "title", "stage", "amount", "region"}
    assert "13812345678" not in minimized[0]["title"]
    assert "secret.example" not in minimized[0]["title"]

    request = build_review_request(
        records, review_mode="engineering", model=DEFAULT_MODEL
    )
    serialized = json.dumps(request, ensure_ascii=False)
    assert "不应发送的联系人" not in serialized
    assert "028-12345678" not in serialized
    assert "secret.example" not in serialized
    sent = json.loads(request["messages"][1]["content"])
    assert set(sent) == {"review_mode", "records"}
    assert set(sent["records"][0]) == {
        "record_id",
        "title",
        "stage",
        "amount",
        "region",
    }
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_excerpt_evidence_is_sent_but_contacts_and_links_are_redacted() -> None:
    evidence = "公告明确写明：本项目采购雇主责任险，保险期限一年。"
    records = [
        {
            "record_id": "E-1",
            "title": "雇主责任保险采购公告",
            "stage": "采购公告",
            "amount": "85000",
            "region": "成都金牛区",
            "公告正文": (
                evidence
                + " 联系人：张三；联系电话：028-12345678；"
                + "手机：13812345678；邮箱 buyer@example.com；"
                + "详情 https://example.com/private "
                + "正文" * 2_000
            ),
            "source_type": "保险类",
            "联系人": "外层联系人也不应发送",
            "详情链接": "https://example.com/raw",
        }
    ]

    minimized = minimize_records(records)
    assert set(minimized[0]) == {
        "record_id",
        "title",
        "stage",
        "amount",
        "region",
        "excerpt",
        "source_type",
    }
    assert minimized[0]["source_type"] == "insurance"
    assert evidence in minimized[0]["excerpt"]
    assert len(minimized[0]["excerpt"]) == MAX_EXCERPT_LENGTH
    assert "张三" not in minimized[0]["excerpt"]
    assert "028-12345678" not in minimized[0]["excerpt"]
    assert "13812345678" not in minimized[0]["excerpt"]
    assert "buyer@example.com" not in minimized[0]["excerpt"]
    assert "example.com" not in minimized[0]["excerpt"]

    request = build_review_request(records, review_mode="mixed")
    serialized = json.dumps(request, ensure_ascii=False)
    assert evidence in serialized
    assert "外层联系人也不应发送" not in serialized
    assert "028-12345678" not in serialized
    assert "example.com" not in serialized
    sent = json.loads(request["messages"][1]["content"])
    assert sent["records"][0]["source_type"] == "insurance"
    assert "excerpt" in sent["records"][0]
    assert "必须优先依据 excerpt" in request["messages"][0]["content"]
    assert "引用一段来自 excerpt 的短证据" in request["messages"][0]["content"]


def test_spaced_and_masked_phone_variants_are_redacted() -> None:
    minimized = minimize_records(
        [
            {
                "record_id": "masked-phone",
                "title": "责任保险采购公告",
                "excerpt": "联 系 电 话：0830-65****5；手 机：138****5678",
                "source_type": "保险",
            }
        ]
    )
    assert "0830-65****5" not in minimized[0]["excerpt"]
    assert "138****5678" not in minimized[0]["excerpt"]


def test_optional_evidence_aliases_and_unknown_source_type_are_safe() -> None:
    aliases = ("excerpt", "内容摘要", "证据摘录")
    for alias in aliases:
        minimized = minimize_records(
            [
                {
                    "record_id": alias,
                    "title": "道路改造工程",
                    alias: "公告证据：施工范围包含道路和排水改造。",
                    "来源类型": "未知内部标签",
                }
            ]
        )
        assert minimized[0]["excerpt"].startswith("公告证据")
        assert "source_type" not in minimized[0]


def test_no_api_key_disables_without_constructing_client() -> None:
    factory_called = False

    def forbidden_factory(**_: Any) -> Any:
        nonlocal factory_called
        factory_called = True
        raise AssertionError("没有密钥时不应创建客户端")

    result = review_opportunities(
        [{"record_id": "1", "title": "安全生产责任保险"}],
        api_key="  ",
        review_mode="insurance",
        client_factory=forbidden_factory,
    )

    assert factory_called is False
    assert result["status"] == "disabled"
    assert result["enabled"] is False
    assert result["reviews"] == [
        {
            "record_id": "1",
            "category": "待判断",
            "decision": "review",
            "reason": "AI 复核未启用：未配置 API Key，请人工复核。",
            "confidence": 0.0,
            "source": "fallback",
        }
    ]


def test_fenced_json_is_parsed_and_normalized() -> None:
    model_payload = {
        "reviews": [
            {
                "record_id": "x-1",
                "category": "责任险",
                "decision": "include",
                "reason": "明确采购雇主责任险",
                "confidence": "95%",
            },
            {
                "record_id": "x-2",
                "category": "无关",
                "decision": "exclude",
                "reason": "保险仅为机械零件名称",
                "confidence": 0.99,
            },
        ]
    }
    fake = _FakeClient(
        lambda _: _response(
            "```json\n" + json.dumps(model_payload, ensure_ascii=False) + "\n```"
        )
    )
    result = review_opportunities(
        [
            {"record_id": "x-1", "title": "雇主责任保险采购"},
            {"record_id": "x-2", "title": "保险孔螺钉采购"},
        ],
        api_key="test-key",
        review_mode="insurance",
        client=fake,
        max_attempts=1,
    )

    assert result["status"] == "ok"
    assert result["reviewed_count"] == 2
    assert result["fallback_count"] == 0
    assert result["reviews"][0]["confidence"] == 0.95
    assert result["reviews"][0]["source"] == "ai"
    assert len(fake.completions.calls) == 1


def test_insurance_response_accepts_multiple_target_categories() -> None:
    payload = {
        "reviews": [
            {
                "record_id": "multi-1",
                "category": "意外险、健康险",
                "decision": "include",
                "reason": "采购员工意外及补充医疗保险",
                "confidence": 0.97,
            }
        ]
    }
    result = validate_review_object(
        payload,
        [{"record_id": "multi-1", "title": "员工意外及补充医疗保险采购"}],
        review_mode="insurance",
    )
    assert result[0]["category"] == "意外险、健康险"


def test_requested_batch_size_is_hard_capped_and_records_are_chunked() -> None:
    fake = _FakeClient(_echo_valid_response)
    records = [
        {"record_id": f"R-{index}", "title": f"道路改造施工 {index}"}
        for index in range(MAX_BATCH_SIZE + 3)
    ]

    result = review_opportunities(
        records,
        api_key="test-key",
        review_mode="engineering",
        batch_size=10_000,
        client=fake,
        max_attempts=1,
    )

    assert result["status"] == "ok"
    assert result["batch_size"] == MAX_BATCH_SIZE
    assert result["batch_count"] == 2
    assert result["reviewed_count"] == MAX_BATCH_SIZE + 3
    sent_sizes = [
        len(json.loads(call["messages"][1]["content"])["records"])
        for call in fake.completions.calls
    ]
    assert sent_sizes == [MAX_BATCH_SIZE, 3]


def test_invalid_protocol_safely_falls_back_instead_of_guessing() -> None:
    bad_payload = {
        "reviews": [
            {
                "record_id": "1",
                "category": "工程直接",
                "decision": "include",
                "reason": "看起来像工程",
                "confidence": 0.8,
            }
        ],
        "unexpected": "not allowed",
    }
    fake = _FakeClient(
        lambda _: _response(json.dumps(bad_payload, ensure_ascii=False))
    )

    result = review_opportunities(
        [{"record_id": "1", "title": "某施工项目"}],
        api_key="test-key",
        review_mode="engineering",
        client=fake,
        max_attempts=1,
    )

    assert result["status"] == "error"
    assert result["reviews"][0]["decision"] == "review"
    assert result["reviews"][0]["category"] == "待判断"
    assert result["reviews"][0]["source"] == "fallback"
    assert "结构化协议" in result["reviews"][0]["reason"]


def test_timeout_is_friendly_and_api_key_is_never_returned() -> None:
    fake = _FakeClient(lambda _: TimeoutError("upstream timed out"))
    secret = "test-secret-must-never-appear"
    result = review_opportunities(
        [{"record_id": "1", "title": "货物运输保险采购"}],
        api_key=secret,
        review_mode="insurance",
        client=fake,
        max_attempts=1,
    )

    assert result["status"] == "error"
    assert result["reviews"][0]["decision"] == "review"
    assert "请求超时" in result["reviews"][0]["reason"]
    assert secret not in json.dumps(result, ensure_ascii=False)


def test_client_factory_receives_config_but_result_exposes_no_key() -> None:
    fake = _FakeClient(_echo_valid_response)
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Any:
        received.update(kwargs)
        return fake

    result = review_opportunities(
        [{"record_id": "1", "title": "道路施工"}],
        api_key="private-key",
        review_mode="engineering",
        base_url="https://compatible.example/v1/",
        model="domestic-model",
        client_factory=factory,
        max_attempts=1,
    )

    assert received == {
        "api_key": "private-key",
        "base_url": "https://compatible.example/v1",
        "timeout": 60.0,
    }
    assert result["status"] == "ok"
    assert result["model"] == "domestic-model"
    assert "private-key" not in json.dumps(result, ensure_ascii=False)
    assert "extra_body" not in fake.completions.calls[0]


def test_protocol_rejects_missing_or_unknown_records() -> None:
    with pytest.raises(AIReviewProtocolError, match="数量"):
        validate_review_object(
            {"reviews": []},
            [{"record_id": "required", "title": "工程"}],
            review_mode="engineering",
        )

    assert parse_json_object('说明文字 {"reviews": []} 结束') == {"reviews": []}
    with pytest.raises(AIReviewProtocolError, match="JSON 对象"):
        parse_json_object("not-json")


def test_defaults_are_deepseek_compatible() -> None:
    assert DEFAULT_MODEL == "deepseek-v4-flash"
    assert DEFAULT_BASE_URL == "https://api.deepseek.com"
