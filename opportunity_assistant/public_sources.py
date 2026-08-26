"""四川省公共资源交易信息网的合法公开信息采集适配器。

只调用官网公开全文检索接口，不登录、不携带身份令牌、不处理验证码，也不尝试
绕过任何技术措施。采集结果保留官方链接与来源标识，供商机助手后续规则筛选。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
import json
import math
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote, urljoin, urlparse, urlunparse

import pandas as pd

from .core import STANDARD_COLUMNS, clean_amount, clean_date


OFFICIAL_HOST = "ggzyjy.sc.gov.cn"
OFFICIAL_ORIGIN = f"https://{OFFICIAL_HOST}"
SEARCH_ENDPOINT = (
    f"{OFFICIAL_ORIGIN}/inteligentsearch/rest/esinteligentsearch/"
    "getFullTextDataNew"
)
SOURCE_PLATFORM = "四川省公共资源交易信息网"
DEFAULT_KEYWORDS = ("险", "工程")
DEFAULT_PAGE_SIZE = 30
MAX_PAGE_SIZE = 50
HARD_MAX_RECORDS_PER_KEYWORD = 2_000
MAX_CONCURRENCY = 2
MAX_DETAIL_CONCURRENCY = 8
MAX_DATE_SPAN_DAYS = 366

PUBLIC_EXTRA_COLUMNS = [
    "公告正文",
    "内容摘要",
    "来源平台",
    "数据来源",
    "官方来源标识",
    "金额口径",
    "金额提取依据",
    "来源分类",
    "报名截止原文",
    "投标截止原文",
    "正文取证状态",
]
PUBLIC_COLUMNS = STANDARD_COLUMNS + PUBLIC_EXTRA_COLUMNS


class PublicSourceError(RuntimeError):
    """公开来源采集的基类异常。"""


class PublicSourceSecurityError(PublicSourceError):
    """请求或响应试图离开允许的官方域名。"""


class PublicSourceRequestError(PublicSourceError):
    """公开接口在有限重试后仍无法访问。"""


class PublicSourceProtocolError(PublicSourceError):
    """公开接口响应不符合已知 JSON 协议。"""


class _NonRetryableRequestError(PublicSourceRequestError):
    """永久性 HTTP 拒绝；不得通过重复请求尝试规避。"""


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "br",
        "p",
        "div",
        "li",
        "tr",
        "table",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        elif lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
        elif lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


class _TargetTextExtractor(HTMLParser):
    """只提取指定 id 元素内的文字，避免把官网导航和页脚混入正文。"""

    _BLOCK_TAGS = _TextExtractor._BLOCK_TAGS

    def __init__(self, target_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.target_id = target_id
        self.parts: list[str] = []
        self._active = False
        self._target_tag = ""
        self._same_tag_depth = 0
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        attributes = {str(key).casefold(): value for key, value in attrs}
        if not self._active and attributes.get("id") == self.target_id:
            self._active = True
            self._target_tag = lowered
            self._same_tag_depth = 1
            return
        if not self._active:
            return
        if lowered == self._target_tag:
            self._same_tag_depth += 1
        if lowered in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        elif lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self._active and tag.casefold() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._active:
            return
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
        elif lowered in self._BLOCK_TAGS:
            self.parts.append("\n")
        if lowered == self._target_tag:
            self._same_tag_depth -= 1
            if self._same_tag_depth <= 0:
                self._active = False

    def handle_data(self, data: str) -> None:
        if self._active and not self._hidden_depth:
            self.parts.append(data)


def clean_html(value: Any) -> str:
    """去除 HTML/高亮标签和脚本，同时尽量保留段落边界。"""

    if value is None:
        return ""
    source = str(value)
    parser = _TextExtractor()
    try:
        parser.feed(source)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        # 面对不完整片段时采用保守回退；不执行或解析任何脚本。
        text = re.sub(r"<[^>]+>", " ", source)
    text = unescape(text).replace("\xa0", " ").replace("\u3000", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_official_detail_text(value: Any) -> str:
    """从四川官网详情页的 ``#newsText`` 容器提取完整公开正文。"""

    if value is None:
        return ""
    parser = _TargetTextExtractor("newsText")
    try:
        parser.feed(str(value))
        parser.close()
    except Exception as exc:
        raise PublicSourceProtocolError("官方详情页 HTML 无法解析") from exc
    text = unescape("".join(parser.parts)).replace("\xa0", " ").replace("\u3000", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_html(value)).strip()


def _parse_date(value: date | datetime | str, *, field: str) -> date:
    parsed = clean_date(value)
    if parsed is None:
        raise ValueError(f"{field} 不是有效日期")
    return parsed


def _validate_date_range(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> tuple[date, date]:
    start = _parse_date(start_date, field="start_date")
    end = _parse_date(end_date, field="end_date")
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if (end - start).days > MAX_DATE_SPAN_DAYS:
        raise ValueError(f"单次公开采集日期跨度不能超过 {MAX_DATE_SPAN_DAYS} 天")
    return start, end


def validate_official_url(value: str, *, base_url: str = OFFICIAL_ORIGIN) -> str:
    """规范化并验证 URL，只允许四川省公共资源交易信息网主域名。"""

    absolute = urljoin(base_url.rstrip("/") + "/", str(value or "").strip())
    parsed = urlparse(absolute)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or (parsed.hostname or "").casefold() != OFFICIAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 80, 443}
    ):
        raise PublicSourceSecurityError("公开来源 URL 不属于允许的四川官方域名")
    # 对官网返回的 http 旧链接统一升级为 https，避免后续明文访问。
    netloc = OFFICIAL_HOST if parsed.port in {None, 80, 443} else parsed.netloc
    return urlunparse(("https", netloc, parsed.path or "/", "", parsed.query, ""))


_AMOUNT_LABELS: tuple[tuple[str, str], ...] = (
    ("预算金额", r"预算金额|采购预算|项目预算|预算价|预算总额"),
    ("最高限价", r"最高投标限价|最高限价|最高响应限价"),
    ("招标控制价", r"招标控制价|采购控制价|控制价"),
    ("投资额", r"标段投资(?:金额|额)?|项目总投资|总投资|投资金额|投资额|工程投资"),
    ("合同估算价", r"合同估算价|估算金额"),
)
_ENGINEERING_AMOUNT_LABELS: tuple[tuple[str, str], ...] = (
    # 工程险线索按底层项目规模判断：有项目/标段投资额时优先采用；
    # 未披露投资额时再回退到控制价、最高限价、预算或合同估算价。
    ("投资额", r"标段投资(?:金额|额)?|项目总投资|总投资|投资金额|投资额|工程投资"),
    ("招标控制价", r"招标控制价|采购控制价|控制价"),
    ("最高限价", r"最高投标限价|最高限价|最高响应限价"),
    ("预算金额", r"预算金额|采购预算|项目预算|预算价|预算总额"),
    ("合同估算价", r"合同估算价|估算金额"),
)
_NUMBER_WITH_UNIT = re.compile(
    r"(?P<number>\d{1,3}(?:[,，]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>万亿元|亿元|万元|万|元)?",
    re.IGNORECASE,
)


def _extract_amount_with_labels(
    text: Any, labels: Sequence[tuple[str, str]]
) -> tuple[float | None, str, str]:
    content = _single_line(text)
    if not content:
        return None, "", ""
    for amount_type, label_pattern in labels:
        for label_match in re.finditer(label_pattern, content, flags=re.IGNORECASE):
            # 仅考察标签后的近邻，避免把远处年份、电话或其他项目金额误配。
            window = content[label_match.end() : label_match.end() + 64]
            window = re.sub(r"^[\s：:为约人民币含税不含税总计共计]*", "", window)
            number_match = _NUMBER_WITH_UNIT.search(window)
            if number_match is None or number_match.start() > 20:
                continue
            raw = number_match.group(0)
            number = clean_amount(raw)
            if number is None or not math.isfinite(number) or number < 0:
                continue
            evidence_start = label_match.start()
            evidence_end = min(len(content), label_match.end() + number_match.end() + 12)
            evidence = content[evidence_start:evidence_end].strip(" ，,；;")
            return float(number), amount_type, evidence
    return None, "", ""


def extract_amount(text: Any) -> tuple[float | None, str, str]:
    """通用金额提取：优先采购预算，再看限价、控制价与投资额。"""

    return _extract_amount_with_labels(text, _AMOUNT_LABELS)


def extract_engineering_amount(text: Any) -> tuple[float | None, str, str]:
    """工程规模金额提取：优先项目/标段投资额，再回退到本次招标金额。"""

    return _extract_amount_with_labels(text, _ENGINEERING_AMOUNT_LABELS)


_CITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("阿坝藏族羌族自治州", "阿坝"),
    ("甘孜藏族自治州", "甘孜"),
    ("凉山彝族自治州", "凉山"),
    ("攀枝花市", "攀枝花"),
    ("成都市", "成都"),
    ("自贡市", "自贡"),
    ("泸州市", "泸州"),
    ("德阳市", "德阳"),
    ("绵阳市", "绵阳"),
    ("广元市", "广元"),
    ("遂宁市", "遂宁"),
    ("内江市", "内江"),
    ("乐山市", "乐山"),
    ("南充市", "南充"),
    ("眉山市", "眉山"),
    ("宜宾市", "宜宾"),
    ("广安市", "广安"),
    ("达州市", "达州"),
    ("雅安市", "雅安"),
    ("巴中市", "巴中"),
    ("资阳市", "资阳"),
    ("阿坝州", "阿坝"),
    ("甘孜州", "甘孜"),
    ("凉山州", "凉山"),
)

_CHENGDU_DISTRICTS = (
    "四川天府新区",
    "成都高新区",
    "天府新区",
    "高新区",
    "龙泉驿区",
    "青白江区",
    "锦江区",
    "青羊区",
    "金牛区",
    "武侯区",
    "成华区",
    "新都区",
    "温江区",
    "双流区",
    "郫都区",
    "新津区",
    "都江堰市",
    "彭州市",
    "邛崃市",
    "崇州市",
    "简阳市",
    "金堂县",
    "大邑县",
    "蒲江县",
)

_INVALID_DISTRICT_PREFIXES = (
    "公共资源",
    "交易中心",
    "项目所在",
    "建设项目",
    "采购项目",
    "招标项目",
)


def detect_sichuan_location(text: Any) -> tuple[str, str]:
    """从标题/正文识别四川市州与紧邻的区县市。"""

    content = _single_line(text)
    if not content:
        return "", ""

    city = ""
    city_match: re.Match[str] | None = None
    for alias, normalized in _CITY_ALIASES:
        match = re.search(re.escape(alias), content)
        if match and (city_match is None or match.start() < city_match.start()):
            city, city_match = normalized, match

    district = ""
    if city_match is not None:
        tail = content[city_match.end() : city_match.end() + 24]
        tail = re.sub(r"^[\s﹒·.、,/\-—_：:]+", "", tail)
        match = re.match(r"([\u4e00-\u9fff]{2,7}?(?:区|县|市))", tail)
        if match:
            candidate = match.group(1)
            if not candidate.startswith(_INVALID_DISTRICT_PREFIXES):
                district = candidate

    # 成都标题常只写区县，使用明确、有限的行政区词表反推市级。
    for candidate in _CHENGDU_DISTRICTS:
        if candidate in content:
            if not city:
                city = "成都"
            if city == "成都" and not district:
                district = candidate
            break

    if not district:
        location_match = re.search(
            r"(?:建设地点|项目地点|项目所在地|采购人地址|地址|位于)[：:\s]*"
            r"(?:四川省)?(?:[\u4e00-\u9fff]{2,8}(?:市|州))?"
            r"([\u4e00-\u9fff]{2,7}?(?:区|县|市))",
            content,
        )
        if location_match:
            district = location_match.group(1)
    return city, district


_RESULT_STAGE_PATTERN = re.compile(
    r"中标(?:结果公告|结果|公告|公示|通知书?|候选人(?:公示)?)|"
    r"成交(?:结果公告|结果|公告|候选人(?:公示)?)|合同(?:公告|公示|签订|履约)|"
    r"流标|废标|终止(?:公告|招标|采购)?|采购失败|招标失败|评标结果|开标记录|"
    r"履约验收|验收公告|结果公告|采购结果|定标",
    re.IGNORECASE,
)


def is_active_opportunity(title: Any, category: Any = "") -> bool:
    """仅明确排除中标、合同、流标等结果类，未知公告保守保留。"""

    material = f"{_single_line(title)} {_single_line(category)}"
    return bool(material.strip()) and _RESULT_STAGE_PATTERN.search(material) is None


def map_announcement_stage(title: Any, category: Any = "") -> str:
    """把官网多种公告名称映射为现有商机助手阶段。"""

    material = f"{_single_line(title)} {_single_line(category)}"
    if _RESULT_STAGE_PATTERN.search(material):
        return "结果类公告"
    if re.search(r"招标计划|采购意向|招标预告|预公示|提前公示", material):
        return "招标预告"
    if re.search(r"更正|变更", material):
        return "信息变更"
    if re.search(r"答疑|补遗|澄清", material):
        return "答疑公告"
    if re.search(r"重新招标|再次招标|二次招标|第二次", material):
        return "重新招标"
    if re.search(r"审批|核准|备案|受理", material):
        return "审批项目"
    return "招标公告"


def _extract_project_number(text: str) -> str:
    labels = r"项目编号|招标编号|采购项目编号|采购编号|项目代码|标段编号"
    stop = r"项目名称|采购项目名称|招标人|采购人|项目业主|预算金额|采购预算|$"
    match = re.search(
        rf"(?:{labels})\s*[：:]\s*(?P<value>.{{2,80}}?)(?=(?:{stop}))",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    value = re.sub(r"\s+", "", match.group("value"))
    value = value.strip("，,。；;：:()（）[]【】")
    return value[:80]


def _extract_labeled_text(text: str, labels: Sequence[str], stops: Sequence[str]) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    stop_pattern = "|".join(re.escape(stop) for stop in stops)
    match = re.search(
        rf"(?:{label_pattern})\s*[：:]\s*(.{{2,100}}?)(?=(?:{stop_pattern})\s*[：:]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" ，,。；;")[:100]


def _extract_deadline_text(text: str, labels: Sequence[str]) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{label_pattern}).{{0,50}}?"
        r"(?P<year>20\d{2})(?:年|[-/.])(?P<month>\d{1,2})(?:月|[-/.])"
        r"(?P<day>\d{1,2})(?:日)?"
        r"(?:\s*(?P<hour>\d{1,2})(?:时|:)(?P<minute>\d{1,2})?(?:分)?)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    value = (
        f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-"
        f"{int(match.group('day')):02d}"
    )
    if match.group("hour") is not None:
        value += f" {int(match.group('hour')):02d}:{int(match.group('minute') or 0):02d}"
    return value


def _extract_deadline(text: str, labels: Sequence[str]) -> date | None:
    value = _extract_deadline_text(text, labels)
    return clean_date(value) if value else None


def _record_category(record: Mapping[str, Any], category_names: Mapping[str, str] | None) -> str:
    direct = _single_line(record.get("categoryname", ""))
    if direct:
        return direct
    category_num = _single_line(record.get("categorynum", ""))
    return _single_line((category_names or {}).get(category_num, ""))


def normalize_public_record(
    record: Mapping[str, Any],
    *,
    keyword: str,
    category_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """把一条官网检索记录规范化为商机助手公共字段。"""

    title = _single_line(record.get("title") or record.get("titlenew") or "")
    body = clean_html(record.get("content") or "")
    if not title:
        raise PublicSourceProtocolError("公开记录缺少标题")
    raw_link = str(record.get("linkurl") or "").strip()
    if not raw_link:
        raise PublicSourceProtocolError("公开记录缺少官方详情链接")
    link = validate_official_url(raw_link)
    category = _record_category(record, category_names)
    amount_extractor = extract_engineering_amount if keyword == "工程" else extract_amount
    amount, amount_type, amount_evidence = amount_extractor(f"{title}\n{body}")
    city, district = detect_sichuan_location(f"{title}\n{body}")
    official_id = _single_line(record.get("id") or record.get("syscollectguid") or "")
    if not official_id:
        official_id = urlparse(link).path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
    published = clean_date(record.get("infodate") or record.get("webdate"))
    compact_body = _single_line(body)

    tenderer = _extract_labeled_text(
        compact_body,
        ("招标人", "采购人", "项目业主"),
        ("联系人", "联系电话", "电话", "招标代理机构", "采购代理机构", "项目名称"),
    )
    agent = _extract_labeled_text(
        compact_body,
        ("招标代理机构", "采购代理机构", "代理机构"),
        ("联系人", "联系电话", "电话", "地址", "项目名称"),
    )
    contact = _extract_labeled_text(
        compact_body,
        ("联系人", "联 系 人"),
        ("联系电话", "电话", "联系方式", "地址", "电子邮件"),
    )
    phone_match = re.search(
        r"(?:联系电话|电话|联系方式)\s*[：:]\s*([0-9*＊\\\-]{5,24})",
        compact_body,
    )
    phone = phone_match.group(1) if phone_match else ""
    registration_deadline_text = _extract_deadline_text(
        compact_body,
        ("报名截止时间", "获取招标文件截止时间", "获取采购文件截止时间"),
    )
    bid_deadline_text = _extract_deadline_text(
        compact_body,
        (
            "投标截止时间",
            "提交投标文件截止时间",
            "响应文件提交截止时间",
            "响应截止时间",
            "投标文件递交的截止时间",
        ),
    )

    normalized: dict[str, Any] = {column: "" for column in STANDARD_COLUMNS}
    normalized.update(
        {
            "关键词": keyword,
            "项目名称": title,
            "信息发布时间": published,
            "项目编号": _extract_project_number(compact_body),
            "发布省份": "四川",
            "发布市级": city,
            "发布区级": district,
            "招标阶段": map_announcement_stage(title, category),
            "报名截止时间": clean_date(registration_deadline_text),
            "投标截止时间": clean_date(bid_deadline_text),
            "招标金额（元）": amount,
            "招标单位": tenderer,
            "招标单位联系人": contact,
            "招标单位联系人电话": phone,
            "代理单位": agent,
            "官网查看地址": link,
            "公告正文": body,
            "内容摘要": compact_body[:500],
            "来源平台": SOURCE_PLATFORM,
            "数据来源": _single_line(record.get("zhuanzai") or SOURCE_PLATFORM),
            "官方来源标识": official_id,
            "金额口径": amount_type,
            "金额提取依据": amount_evidence,
            "来源分类": category or _single_line(record.get("categorynum", "")),
            "报名截止原文": registration_deadline_text,
            "投标截止原文": bid_deadline_text,
            "正文取证状态": "检索摘要",
        }
    )
    return normalized


def fetch_official_detail_text(
    url: str,
    *,
    session: Any | None = None,
    timeout: float = 15.0,
    max_retries: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """读取一条无需登录的四川官网详情页并返回完整正文。"""

    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if not 0 <= int(max_retries) <= 3:
        raise ValueError("max_retries 必须在 0 到 3 之间")
    target = validate_official_url(url)
    if session is None:
        import requests

        request_get = requests.get
    else:
        request_get = session.get

    last_error: Exception | None = None
    for attempt in range(int(max_retries) + 1):
        try:
            response = request_get(
                target,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "Referer": f"{OFFICIAL_ORIGIN}/",
                    "User-Agent": "OpportunityAssistant/1.0 (public official detail; low-rate)",
                },
                timeout=float(timeout),
            )
            response_url = getattr(response, "url", target) or target
            validate_official_url(str(response_url))
            status = int(getattr(response, "status_code", 200))
            if status == 429 or 500 <= status <= 599:
                raise PublicSourceRequestError(f"官方详情页暂时不可用（HTTP {status}）")
            if not 200 <= status <= 299:
                raise _NonRetryableRequestError(f"官方详情页拒绝请求（HTTP {status}）")
            encoding = str(getattr(response, "encoding", "") or "").casefold()
            if not encoding or encoding in {"iso-8859-1", "latin-1"}:
                try:
                    response.encoding = "utf-8"
                except Exception:
                    pass
            html = getattr(response, "text", "")
            if not html:
                raw = getattr(response, "content", b"")
                html = bytes(raw).decode("utf-8", errors="replace") if raw else ""
            detail = extract_official_detail_text(html)
            if not detail:
                raise PublicSourceProtocolError("官方详情页未找到公开正文容器")
            return detail
        except (PublicSourceSecurityError, _NonRetryableRequestError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= int(max_retries):
                break
            sleep(min(2.0, 0.4 * (2**attempt)))
    raise PublicSourceRequestError(
        f"官方详情页读取失败，已重试 {int(max_retries)} 次：{last_error}"
    ) from last_error


def _enrich_row_from_detail(frame: pd.DataFrame, index: Any, detail: str) -> None:
    title = _single_line(frame.at[index, "项目名称"] if "项目名称" in frame else "")
    material = f"{title}\n{detail}"
    frame.at[index, "公告正文"] = detail
    frame.at[index, "内容摘要"] = _single_line(detail)[:500]
    frame.at[index, "正文取证状态"] = "完整正文"

    keyword = _single_line(frame.at[index, "关键词"] if "关键词" in frame else "")
    amount_extractor = extract_engineering_amount if keyword == "工程" else extract_amount
    amount, amount_type, amount_evidence = amount_extractor(material)
    if amount is not None:
        frame.at[index, "招标金额（元）"] = amount
        frame.at[index, "金额口径"] = amount_type
        frame.at[index, "金额提取依据"] = amount_evidence

    city, district = detect_sichuan_location(material)
    if city and not _single_line(frame.at[index, "发布市级"]):
        frame.at[index, "发布市级"] = city
    if district and not _single_line(frame.at[index, "发布区级"]):
        frame.at[index, "发布区级"] = district

    project_number = _extract_project_number(_single_line(detail))
    if project_number and not _single_line(frame.at[index, "项目编号"]):
        frame.at[index, "项目编号"] = project_number
    compact = _single_line(detail)
    tenderer = _extract_labeled_text(
        compact,
        ("招标人", "采购人", "项目业主"),
        ("联系人", "联系电话", "电话", "招标代理机构", "采购代理机构", "项目名称"),
    )
    agent = _extract_labeled_text(
        compact,
        ("招标代理机构", "采购代理机构", "代理机构"),
        ("联系人", "联系电话", "电话", "地址", "项目名称"),
    )
    contact = _extract_labeled_text(
        compact,
        ("联系人", "联 系 人"),
        ("联系电话", "电话", "联系方式", "地址", "电子邮件"),
    )
    phone_match = re.search(
        r"(?:联系电话|电话|联系方式)\s*[：:]\s*([0-9*＊\\\-]{5,24})",
        compact,
    )
    for column, value in (
        ("招标单位", tenderer),
        ("代理单位", agent),
        ("招标单位联系人", contact),
        ("招标单位联系人电话", phone_match.group(1) if phone_match else ""),
    ):
        if value and not _single_line(frame.at[index, column]):
            frame.at[index, column] = value
    for column, raw_column, labels in (
        (
            "报名截止时间",
            "报名截止原文",
            ("报名截止时间", "获取招标文件截止时间", "获取采购文件截止时间"),
        ),
        (
            "投标截止时间",
            "投标截止原文",
            (
                "投标截止时间",
                "提交投标文件截止时间",
                "响应文件提交截止时间",
                "响应截止时间",
                "投标文件递交的截止时间",
            ),
        ),
    ):
        raw_deadline = _extract_deadline_text(compact, labels)
        parsed = clean_date(raw_deadline)
        if parsed is not None:
            frame.at[index, column] = parsed
            frame.at[index, raw_column] = raw_deadline


def enrich_public_dataframe(
    frame: pd.DataFrame,
    *,
    candidate_mask: Any | None = None,
    session: Any | None = None,
    timeout: float = 15.0,
    max_retries: int = 1,
    request_delay: float = 0.05,
    max_workers: int = MAX_DETAIL_CONCURRENCY,
    max_records: int = 1_000,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅为规则候选读取官网详情，补齐金额、正文、地区和截止日期。"""

    result = frame.copy(deep=True)
    if "官网查看地址" not in result.columns:
        raise ValueError("公开记录缺少官网查看地址")
    if max_records < 1:
        raise ValueError("max_records 必须大于 0")
    if request_delay < 0:
        raise ValueError("request_delay 不能为负数")
    if max_workers < 1:
        raise ValueError("max_workers 必须大于 0")
    if candidate_mask is None:
        mask = pd.Series(True, index=result.index)
    elif isinstance(candidate_mask, pd.Series):
        mask = candidate_mask.reindex(result.index, fill_value=False).astype(bool)
    else:
        values = list(candidate_mask)
        if len(values) != len(result):
            raise ValueError("candidate_mask 长度与数据行数不一致")
        mask = pd.Series(values, index=result.index).fillna(False).astype(bool)

    selected_indices = list(result.index[mask])[: int(max_records)]
    if "正文取证状态" not in result.columns:
        result["正文取证状态"] = "检索摘要"
    result.loc[selected_indices, "正文取证状态"] = "待读取"
    url_to_indices: dict[str, list[Any]] = {}
    invalid_url_count = 0
    for index in selected_indices:
        try:
            url = validate_official_url(str(result.at[index, "官网查看地址"] or ""))
        except (PublicSourceSecurityError, ValueError):
            invalid_url_count += 1
            result.at[index, "正文取证状态"] = "链接无效"
            continue
        url_to_indices.setdefault(url, []).append(index)

    started = time.monotonic()
    successes: dict[str, str] = {}
    failures: dict[str, str] = {}

    def fetch(url: str) -> tuple[str, str]:
        detail = fetch_official_detail_text(
            url,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            sleep=sleep,
        )
        if request_delay:
            sleep(request_delay)
        return url, detail

    urls = list(url_to_indices)
    workers = 1 if session is not None else min(int(max_workers), MAX_DETAIL_CONCURRENCY)
    if workers == 1:
        for url in urls:
            try:
                key, detail = fetch(url)
                successes[key] = detail
            except PublicSourceError as exc:
                failures[url] = str(exc)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sc-detail") as pool:
            futures = {pool.submit(fetch, url): url for url in urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    key, detail = future.result()
                    successes[key] = detail
                except PublicSourceError as exc:
                    failures[url] = str(exc)

    success_rows = 0
    for url, detail in successes.items():
        for index in url_to_indices[url]:
            _enrich_row_from_detail(result, index, detail)
            success_rows += 1
    for url in failures:
        for index in url_to_indices[url]:
            result.at[index, "正文取证状态"] = "读取失败"

    stats = {
        "candidate_count": int(mask.sum()),
        "attempted_row_count": int(len(selected_indices)),
        "attempted_url_count": int(len(urls)),
        "success_row_count": int(success_rows),
        "success_url_count": int(len(successes)),
        "failure_url_count": int(len(failures)),
        "invalid_url_count": int(invalid_url_count),
        "truncated": bool(int(mask.sum()) > len(selected_indices)),
        "duration_seconds": round(time.monotonic() - started, 3),
        "last_error": failures[next(reversed(failures))] if failures else "",
    }
    return result.reindex(columns=PUBLIC_COLUMNS), stats


def _empty_public_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PUBLIC_COLUMNS)


def _records_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    values = list(records)
    if not values:
        return _empty_public_frame()
    frame = pd.DataFrame.from_records(values)
    for column in PUBLIC_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame = frame.reindex(columns=PUBLIC_COLUMNS)
    return frame.reset_index(drop=True)


def _coerce_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PublicSourceProtocolError("公开接口顶层响应不是 JSON 对象")
    result = payload.get("result", payload)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:
            raise PublicSourceProtocolError("公开接口 result 不是有效 JSON") from exc
    if not isinstance(result, Mapping):
        raise PublicSourceProtocolError("公开接口缺少 result 对象")
    return dict(result)


class SichuanPublicSourceClient:
    """低并发、有限分页的四川官方公开检索客户端。"""

    def __init__(
        self,
        *,
        session: Any | None = None,
        timeout: float = 15.0,
        max_retries: int = 2,
        page_size: int = DEFAULT_PAGE_SIZE,
        request_delay: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        if not 1 <= int(page_size) <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size 必须在 1 到 {MAX_PAGE_SIZE} 之间")
        if not 0 <= int(max_retries) <= 5:
            raise ValueError("max_retries 必须在 0 到 5 之间")
        if request_delay < 0:
            raise ValueError("request_delay 不能为负数")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.page_size = int(page_size)
        self.request_delay = float(request_delay)
        self.sleep = sleep
        self.request_count = 0
        self.retry_count = 0

    @staticmethod
    def _payload(
        keyword: str,
        start: date,
        end: date,
        *,
        offset: int,
        page_size: int,
    ) -> dict[str, Any]:
        return {
            "token": "",
            "pn": int(offset),
            "rn": int(page_size),
            "sdt": f"{start.isoformat()} 00:00:00",
            "edt": f"{end.isoformat()} 23:59:59",
            "wd": quote(keyword, safe=""),
            "inc_wd": "",
            "exc_wd": "",
            # 先按标题召回，避免“工程”仅出现在正文机构名称时产生数千条误命中；
            # 搜索结果仍携带摘要，规则候选随后再读取官网详情正文。
            "fields": "title",
            "cnum": "",
            "sort": '{"infodate":0}',
            "ssort": "title",
            "cl": 10_000,
            "terminal": "",
            "condition": None,
            "time": None,
            "highlights": "title",
            "statistics": None,
            "unionCondition": None,
            "accuracy": "",
            "noParticiple": "0",
            "searchRange": None,
        }

    def _post_page(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        validate_official_url(SEARCH_ENDPOINT)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.request_count += 1
                response = self.session.post(
                    SEARCH_ENDPOINT,
                    json=dict(payload),
                    headers={
                        "Accept": "application/json, text/plain, */*",
                        "Content-Type": "application/json;charset=UTF-8",
                        "Referer": f"{OFFICIAL_ORIGIN}/search/fullsearch.html",
                        "User-Agent": "OpportunityAssistant/1.0 (public official search; low-rate)",
                    },
                    timeout=self.timeout,
                )
                response_url = getattr(response, "url", SEARCH_ENDPOINT) or SEARCH_ENDPOINT
                validate_official_url(str(response_url))
                status = int(getattr(response, "status_code", 200))
                if status == 429 or 500 <= status <= 599:
                    raise PublicSourceRequestError(f"官方接口暂时不可用（HTTP {status}）")
                if not 200 <= status <= 299:
                    raise _NonRetryableRequestError(
                        f"官方接口拒绝请求（HTTP {status}）"
                    )
                try:
                    body = response.json()
                except Exception as exc:
                    raise PublicSourceProtocolError("官方接口未返回有效 JSON") from exc
                return _coerce_result(body)
            except (PublicSourceSecurityError, _NonRetryableRequestError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self.retry_count += 1
                self.sleep(min(4.0, 0.5 * (2**attempt)))
        raise PublicSourceRequestError(
            f"官方公开接口请求失败，已重试 {self.max_retries} 次：{last_error}"
        ) from last_error

    def search_keyword(
        self,
        keyword: str,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
        *,
        max_records: int = 500,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """分页检索一个关键词，返回活跃公告和采集统计。"""

        keyword = str(keyword or "").strip()
        if keyword not in DEFAULT_KEYWORDS:
            raise ValueError("四川公开源适配器仅允许关键词“险”或“工程”")
        start, end = _validate_date_range(start_date, end_date)
        if int(max_records) < 1:
            raise ValueError("max_records 必须大于 0")
        effective_limit = min(int(max_records), HARD_MAX_RECORDS_PER_KEYWORD)

        accepted: list[dict[str, Any]] = []
        seen: set[str] = set()
        api_total = 0
        fetched_count = 0
        excluded_result_count = 0
        invalid_count = 0
        pages = 0
        offset = 0

        while fetched_count < effective_limit:
            remaining = effective_limit - fetched_count
            current_page_size = min(self.page_size, remaining)
            payload = self._payload(
                keyword,
                start,
                end,
                offset=offset,
                page_size=current_page_size,
            )
            result = self._post_page(payload)
            pages += 1
            try:
                api_total = max(api_total, int(result.get("totalcount") or 0))
            except (TypeError, ValueError):
                pass
            raw_records = result.get("records") or []
            if not isinstance(raw_records, list):
                raise PublicSourceProtocolError("官方接口 records 不是数组")
            if not raw_records:
                break
            category_names = {
                str(item.get("categorynum") or ""): _single_line(item.get("categoryname") or "")
                for item in (result.get("categorys") or [])
                if isinstance(item, Mapping)
            }

            for raw_record in raw_records[:remaining]:
                fetched_count += 1
                if not isinstance(raw_record, Mapping):
                    invalid_count += 1
                    continue
                title = raw_record.get("title") or raw_record.get("titlenew") or ""
                category = _record_category(raw_record, category_names)
                if not is_active_opportunity(title, category):
                    excluded_result_count += 1
                    continue
                try:
                    normalized = normalize_public_record(
                        raw_record,
                        keyword=keyword,
                        category_names=category_names,
                    )
                except (PublicSourceProtocolError, PublicSourceSecurityError, ValueError):
                    invalid_count += 1
                    continue
                dedupe_key = str(normalized["官方来源标识"] or normalized["官网查看地址"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                accepted.append(normalized)

            offset += len(raw_records)
            if len(raw_records) < current_page_size or (api_total and offset >= api_total):
                break
            if fetched_count < effective_limit and self.request_delay:
                self.sleep(self.request_delay)

        frame = _records_to_frame(accepted)
        stats = {
            "keyword": keyword,
            "api_total": int(api_total),
            "fetched_count": int(fetched_count),
            "active_count": int(len(frame)),
            "excluded_result_count": int(excluded_result_count),
            "invalid_count": int(invalid_count),
            "page_count": int(pages),
            "request_count": int(self.request_count),
            "retry_count": int(self.retry_count),
            "requested_limit": int(max_records),
            "effective_limit": int(effective_limit),
            "truncated": bool(api_total > effective_limit),
            "error": "",
        }
        return frame, stats


def _merge_public_frames(frames: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return _empty_public_frame(), 0
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    keys = combined["官方来源标识"].fillna("").astype(str)
    fallback = combined["官网查看地址"].fillna("").astype(str)
    keys = keys.where(keys.str.strip().ne(""), fallback)
    duplicate_count = int(keys.duplicated(keep="first").sum())
    combined = combined.loc[~keys.duplicated(keep="first")].copy()
    combined = combined.reindex(columns=PUBLIC_COLUMNS)
    if "信息发布时间" in combined:
        combined = combined.sort_values(
            ["信息发布时间", "项目名称"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        )
    return combined.reset_index(drop=True), duplicate_count


def collect_sichuan_public_opportunities(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    max_records_per_keyword: int = 500,
    *,
    session: Any | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = 15.0,
    max_retries: int = 2,
    request_delay: float = 0.2,
    max_workers: int = MAX_CONCURRENCY,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """采集“险/工程”公开记录，返回 insurance/engineering/combined/stats。

    注入 ``session`` 时为保证测试或调用方会话安全，两个关键词顺序执行；正常
    网络模式最多同时执行两个关键词，每个关键词内部仍严格顺序分页。
    """

    start, end = _validate_date_range(start_date, end_date)
    if int(max_records_per_keyword) < 1:
        raise ValueError("max_records_per_keyword 必须大于 0")
    if int(max_workers) < 1:
        raise ValueError("max_workers 必须大于 0")
    workers = min(int(max_workers), MAX_CONCURRENCY)
    started = time.monotonic()

    def run(keyword: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        client = SichuanPublicSourceClient(
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            page_size=page_size,
            request_delay=request_delay,
            sleep=sleep,
        )
        try:
            return client.search_keyword(
                keyword,
                start,
                end,
                max_records=max_records_per_keyword,
            )
        except PublicSourceError as exc:
            return _empty_public_frame(), {
                "keyword": keyword,
                "api_total": 0,
                "fetched_count": 0,
                "active_count": 0,
                "excluded_result_count": 0,
                "invalid_count": 0,
                "page_count": 0,
                "request_count": int(client.request_count),
                "retry_count": int(client.retry_count),
                "requested_limit": int(max_records_per_keyword),
                "effective_limit": min(
                    int(max_records_per_keyword), HARD_MAX_RECORDS_PER_KEYWORD
                ),
                "truncated": False,
                "error": str(exc),
            }

    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    # 外部注入的 requests.Session 未必线程安全，测试与集成模式坚持串行。
    if session is not None or workers == 1:
        for keyword in DEFAULT_KEYWORDS:
            results[keyword] = run(keyword)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sc-public") as pool:
            futures = {pool.submit(run, keyword): keyword for keyword in DEFAULT_KEYWORDS}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    insurance, insurance_stats = results["险"]
    engineering, engineering_stats = results["工程"]
    combined, overlap_count = _merge_public_frames((insurance, engineering))
    stats = {
        "source": SOURCE_PLATFORM,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "keywords": {"险": insurance_stats, "工程": engineering_stats},
        "insurance_count": int(len(insurance)),
        "engineering_count": int(len(engineering)),
        "combined_count": int(len(combined)),
        "keyword_overlap_count": int(overlap_count),
        "request_count": int(
            insurance_stats.get("request_count", 0)
            + engineering_stats.get("request_count", 0)
        ),
        "retry_count": int(
            insurance_stats.get("retry_count", 0)
            + engineering_stats.get("retry_count", 0)
        ),
        "excluded_result_count": int(
            insurance_stats.get("excluded_result_count", 0)
            + engineering_stats.get("excluded_result_count", 0)
        ),
        "duration_seconds": round(time.monotonic() - started, 3),
        "has_errors": bool(insurance_stats.get("error") or engineering_stats.get("error")),
    }
    return {
        "insurance": insurance,
        "engineering": engineering,
        "combined": combined,
        "stats": stats,
    }


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "HARD_MAX_RECORDS_PER_KEYWORD",
    "MAX_CONCURRENCY",
    "MAX_DETAIL_CONCURRENCY",
    "OFFICIAL_HOST",
    "OFFICIAL_ORIGIN",
    "PUBLIC_COLUMNS",
    "PUBLIC_EXTRA_COLUMNS",
    "PublicSourceError",
    "PublicSourceProtocolError",
    "PublicSourceRequestError",
    "PublicSourceSecurityError",
    "SEARCH_ENDPOINT",
    "SOURCE_PLATFORM",
    "SichuanPublicSourceClient",
    "clean_html",
    "collect_sichuan_public_opportunities",
    "detect_sichuan_location",
    "enrich_public_dataframe",
    "extract_amount",
    "extract_engineering_amount",
    "extract_official_detail_text",
    "fetch_official_detail_text",
    "is_active_opportunity",
    "map_announcement_stage",
    "normalize_public_record",
    "validate_official_url",
]
