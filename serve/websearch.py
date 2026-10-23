"""Anthropic's web_search server tool, run here against a SearXNG instance.

A Messages request that offers {"type": "web_search_20250305", "name": "web_search", ...} gets the tool as a function
with one `query` parameter.  When the model calls it, the server queries SearXNG's JSON API, gives the model the
results as the tool's answer and lets it continue; the client sees the call as a server_tool_use block and the
results as a web_search_tool_result block, as from Anthropic's own API (Claude Code's WebSearch reads those).

The run config's `"web_search"` block: {"searxng": "http://127.0.0.1:8888", "max_results": 10, "timeout_s": 15,
"max_uses": 8}.  Without it the tool is offered as before: a call the client has to answer itself."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULTS = {"max_results": 10, "timeout_s": 15.0, "max_uses": 8}
DESCRIPTION = ("Search the web. Returns the title, URL and a snippet of each result. Use it for current events, "
               "recent releases and anything past your training data.")
SCHEMA = {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}},
          "required": ["query"]}


def is_server_tool(tool) -> bool:
    return isinstance(tool, dict) and str(tool.get("type", "")).startswith("web_search_")


def settings_from(cfg: dict) -> dict | None:
    """The run config's optional `"web_search"` block, checked; None when there is none."""
    block = cfg.get("web_search")
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("searxng"), str) or not block["searxng"]:
        raise SystemExit('[strata] config web_search: expected {"searxng": "http://host:port", ...}')
    out = {**DEFAULTS, "searxng": block["searxng"].rstrip("/")}
    for key, value in block.items():
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "searxng":
            continue
        if key == "timeout_s":
            if not number or value <= 0:
                raise SystemExit(f"[strata] config web_search.{key}={value!r}: expected a number of seconds > 0")
            out[key] = float(value)
        elif key in ("max_results", "max_uses"):
            if not number or value != int(value) or value < 1:
                raise SystemExit(f"[strata] config web_search.{key}={value!r}: expected a positive integer")
            out[key] = int(value)
        else:
            print(f"[strata] config web_search.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def _host_matches(url: str, domains) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    for d in domains:
        d = str(d).lower().strip().removeprefix("https://").removeprefix("http://").split("/")[0]
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def search(settings: dict, query: str, allowed=None, blocked=None) -> dict:
    """-> {"ok": True, "results": [{"url", "title", "content", "page_age"}], "ms"} or {"ok": False, "error_code",
    "message", "ms"}.  allowed_domains / blocked_domains filter the results; with allowed_domains the query also
    asks SearXNG for those sites only."""
    t0 = time.monotonic()
    q = query
    if allowed:
        q += " " + " OR ".join(f"site:{d}" for d in allowed)
    url = settings["searxng"] + "/search?" + urllib.parse.urlencode({"q": q, "format": "json"})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "strata", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=settings["timeout_s"]) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "error_code": "unavailable", "message": f"the search failed: {e}",
                "ms": int((time.monotonic() - t0) * 1000)}
    results, seen = [], set()
    for item in data.get("results") or []:
        link = item.get("url")
        if not isinstance(link, str) or not link.startswith(("http://", "https://")) or link in seen:
            continue
        if allowed and not _host_matches(link, allowed):
            continue
        if blocked and _host_matches(link, blocked):
            continue
        seen.add(link)
        results.append({"url": link, "title": str(item.get("title") or link),
                        "content": str(item.get("content") or ""),
                        "page_age": item.get("publishedDate") or None})
        if len(results) >= settings["max_results"]:
            break
    return {"ok": True, "results": results, "ms": int((time.monotonic() - t0) * 1000)}


def model_text(r: dict) -> str:
    """What the model reads as the tool's answer."""
    if not r["ok"]:
        return f"error: {r['message']}"
    if not r["results"]:
        return "No results."
    return "\n\n".join(f"[{i}] {x['title']}\nURL: {x['url']}" + (f"\nDate: {x['page_age']}" if x["page_age"] else "") +
                       (f"\n{x['content']}" if x["content"] else "") for i, x in enumerate(r["results"], 1))


def block_content(r: dict):
    """The web_search_tool_result block's content: the results, or Anthropic's error object."""
    if not r["ok"]:
        return {"type": "web_search_tool_result_error", "error_code": r["error_code"]}
    return [{"type": "web_search_result", "url": x["url"], "title": x["title"], "page_age": x["page_age"],
             "encrypted_content": base64.b64encode(json.dumps(
                 {"content": x["content"]}, ensure_ascii=False).encode()).decode()} for x in r["results"]]


def from_block(content) -> str:
    """A web_search_tool_result block a client sends back in the history, as the model read it."""
    if isinstance(content, dict):
        return f"error: {content.get('error_code', 'unavailable')}"
    results = []
    for x in content or []:
        if not isinstance(x, dict):
            continue
        snippet = ""
        try:
            snippet = json.loads(base64.b64decode(x.get("encrypted_content") or "")).get("content", "")
        except (ValueError, TypeError, AttributeError):
            pass
        results.append({"url": x.get("url", ""), "title": x.get("title", ""), "content": snippet or "",
                        "page_age": x.get("page_age")})
    return model_text({"ok": True, "results": results})
