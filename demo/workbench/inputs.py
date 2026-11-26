"""Single-asset inputs. No network connection is ever made to a submitted IP."""

import csv
import io
import ipaddress
import json
import os
from itertools import islice
from pathlib import Path
from urllib.parse import quote

import requests

from retrieval_contract import has_observation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "config/perspective_info.json").read_text())
INDEX_COLS = json.loads((ROOT / "config/perspective_name.json").read_text())
FIELDS = {col for cols in INDEX_COLS.values() for col in cols} | {col for view in CONFIG.values() for col in view["cols"]} | {
    "http-info", "hw-versions", "cert-fingerprints", "http-favicon-hashes", "loc-info",
    "loc-latitude", "loc-longitude", "loc-continent", "loc-country", "loc-country_code",
    "loc-province", "loc-city", "loc-postal_code", "loc-timezone",
}
LIMIT = 2 * 1024 * 1024
csv.field_size_limit(LIMIT)


class InputError(ValueError):
    pass


def address(value):
    try:
        value = str(value).strip()
        if "%" not in value:
            return str(ipaddress.ip_address(value))
    except ValueError:
        pass
    raise InputError("请输入单个有效 IPv4 或 IPv6 地址，不接受 URL、网段或地址列表。")


def _rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("result", "hits", "fingerprint"):
            if key in obj and isinstance(obj[key], (dict, list)):
                return _rows(obj[key])
        return [obj]
    raise InputError("JSON 必须是指纹对象、记录数组或 Censys 主机记录。")


def _payload(row):
    row = row.get("host_v1", row)
    if not isinstance(row, dict):
        raise InputError("无效的主机记录。")
    body = row.get("resource", row)
    if not isinstance(body, dict):
        raise InputError("无效的 resource 字段。")
    raw = "resource" in row or any(k in body for k in ("services", "autonomous_system", "operating_system"))
    return body, raw


def _omit_nulls(value):
    if isinstance(value, list):
        return [_omit_nulls(v) for v in value if v is not None]
    if isinstance(value, dict):
        return {k: _omit_nulls(v) for k, v in value.items() if v is not None}
    return value


def fingerprint(ip, obj, *, require_ip=False):
    rows = _rows(obj)
    match = None
    for row in rows:
        if not isinstance(row, dict):
            raise InputError("文件必须包含对象记录。")
        body, raw = _payload(row)
        if body.get("ip") not in (None, ""):
            if address(body["ip"]) != ip:
                continue
        elif require_ip or len(rows) != 1:
            continue
        if match is not None:
            raise InputError("同一 IP 有多条记录，请选择一个观测时间后上传。")
        match = body, raw
    if match is None:
        raise InputError("没有与输入 IP 一致的记录。")
    body, raw = match
    if raw:
        from acquire_data import CensysData
        try:
            body = CensysData.extract_features_from_host({"resource": _omit_nulls(body)})
        except (AttributeError, KeyError, TypeError, RecursionError) as exc:
            raise InputError("Censys 主机记录格式错误，无法按项目字段提取。") from exc
    result = {"ip": ip}
    for key in sorted(FIELDS & body.keys()):
        value = body[key]
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        if has_observation(value):
            result[key] = str(value)
    if len(result) == 1:
        raise InputError("没有可识别的指纹字段；仅有 IP 或设备标签不能用于识别。")
    if max(map(len, result.values())) > 16000 or sum(map(len, result.values())) > 64000:
        raise InputError("为限制共享 CPU 内存，单个字段限 16000 字符、每条指纹合计限 64000 字符。"
                         "请精简过长的网页正文等字段后上传。")
    return result


def uploaded(ip, name, content):
    if not content or len(content) > LIMIT:
        raise InputError("请选择非空文件，大小不能超过 2 MiB。")
    try:
        text = content.decode("utf-8-sig")
        suffix = Path(name).suffix.lower()
        if suffix == ".json":
            return fingerprint(ip, json.loads(text))
        if suffix == ".csv":
            rows = list(islice(csv.DictReader(io.StringIO(text, newline=""), strict=True), 4097))
            if len(rows) > 4096:
                raise InputError("单次最多读取 4096 行 CSV，请先筛选目标 IP。")
            return fingerprint(ip, rows)
    except (UnicodeError, json.JSONDecodeError, csv.Error, RecursionError) as exc:
        raise InputError("无法读取文件，请使用 UTF-8 编码的 JSON 或带表头的 CSV。") from exc
    raise InputError("仅支持 .json 或 .csv 文件。")


def credentials():
    path = ROOT / "search_config.json"
    try:
        platform = json.loads(path.read_text())["censys"]["platform"]
    except (OSError, KeyError, TypeError, ValueError):
        platform = {}
    return (os.getenv("CENSYS_PERSONAL_ACCESS_TOKEN") or platform.get("personal_access_token") or "",
            os.getenv("CENSYS_ORG_ID") or platform.get("org_id") or "")


def censys(ip, token="", org=""):
    addr = ipaddress.ip_address(ip)
    if not addr.is_global or addr.is_multicast:
        raise InputError("Censys 只接受公网单播 IP；内网设备请上传指纹。")
    saved_token, saved_org = credentials()
    # A user-supplied PAT must not be charged to an unrelated server organization.
    org = org or (saved_org if not token else "")
    token = token or saved_token
    if not token:
        raise InputError("未配置 Censys PAT。请在本次查询中填写，或改用上传指纹。")
    url = "https://api.platform.censys.io/v3/global/asset/host/" + quote(ip, safe="")
    try:
        with requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                          params={"organization_id": org} if org else {}, timeout=(4, 18),
                          stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                hints = {401: "凭据无效", 403: "权限或额度不足", 404: "未收录该 IP", 429: "达到速率限制"}
                raise InputError(f"Censys {response.status_code}：{hints.get(response.status_code, '查询失败')}。")
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > LIMIT:
                    raise InputError("该主机记录超出 2 MiB，建议筛选字段后以文件上传。")
        return fingerprint(ip, json.loads(data), require_ip=True)
    except (requests.RequestException, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputError("无法读取 Censys 响应，请检查网络或稍后重试。") from exc
