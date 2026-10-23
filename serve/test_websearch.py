"""serve/test_websearch.py - Anthropic's web_search server tool run against a fake SearXNG, with the mock engine.

    python -m unittest serve.test_websearch -v
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import websearch  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 8192
RESULTS = [
    {"url": "https://example.org/strata", "title": "Strata engine", "content": "Runs big MoE models.",
     "publishedDate": "2026-10-01T00:00:00"},
    {"url": "https://spam.example.com/x", "title": "Spam", "content": "buy"},
    {"url": "https://example.org/strata", "title": "Duplicate", "content": "again"},
    {"url": "ftp://example.net/f", "title": "Not the web", "content": ""},
    {"url": "https://docs.example.net/guide", "title": "Guide", "content": ""},
]
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}


class FakeSearxng:
    def __init__(self):
        self.queries = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                outer.queries.append((q.get("q", [""])[0], q.get("format", [""])[0]))
                body = json.dumps({"query": q.get("q", [""])[0], "results": RESULTS}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class SearchingEngine(MockEngine):
    """Calls web_search until the prompt holds `searches` results, then answers.  A prompt that already opens the
    call (a forced tool_choice) gets only its arguments."""
    ANSWER = "Strata runs big MoE models."

    def __init__(self, tok, searches=1, query="strata engine"):
        super().__init__(tok, "x", max_context=CTX)
        self.searches, self.query = searches, query
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(self.tok.decode(ids))
        prompt = self.prompts[-1]
        args = f"<parameter=query>\n{self.query}\n</parameter>\n</function>\n</tool_call>"
        think = "" if prompt.endswith("</think>\n\n") else "</think>\n\n"
        if prompt.endswith("<function=web_search>\n"):
            text = args
        elif prompt.count("<tool_response>") >= self.searches:
            text = think + self.ANSWER
        else:
            text = think + "Searching.\n\n<tool_call>\n<function=web_search>\n" + args
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class ServerTool(unittest.TestCase):
    def setUp(self):
        self.searx = FakeSearxng()
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.searx.close()

    def start(self, web_search=True, **kw):
        tok = ByteTokenizer()
        self.engine = SearchingEngine(tok, **kw)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if web_search:
            self.svc.web_search = websearch.settings_from({"web_search": {"searxng": self.searx.url + "/",
                                                                          "max_results": 2}})
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, stream=True, **body):
        body = {"model": "m", "max_tokens": 4096, "messages": [{"role": "user", "content": "what is strata?"}],
                "tools": [WEB_SEARCH], "stream": stream, **body}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    @staticmethod
    def events(text):
        return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]

    def test_search_then_answer_streamed(self):
        self.start()
        code, text = self.post(tools=[{**WEB_SEARCH, "blocked_domains": ["spam.example.com"]}])
        self.assertEqual(code, 200, text)
        evs = self.events(text)
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
        self.assertEqual([b["type"] for b in starts if b["type"] != "thinking"],
                         ["text", "server_tool_use", "web_search_tool_result", "text"])
        use = next(b for b in starts if b["type"] == "server_tool_use")
        result = next(b for b in starts if b["type"] == "web_search_tool_result")
        self.assertTrue(use["id"].startswith("srvtoolu_"))
        self.assertEqual(result["tool_use_id"], use["id"])
        self.assertEqual([(r["title"], r["url"]) for r in result["content"]],
                         [("Strata engine", "https://example.org/strata"), ("Guide", "https://docs.example.net/guide")])
        snippet = json.loads(base64.b64decode(result["content"][0]["encrypted_content"]))["content"]
        self.assertEqual(snippet, "Runs big MoE models.")
        args = "".join(e["delta"]["partial_json"] for e in evs if e["type"] == "content_block_delta"
                       and e["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(args), {"query": "strata engine"})
        text_out = "".join(e["delta"]["text"] for e in evs if e["type"] == "content_block_delta"
                           and e["delta"]["type"] == "text_delta")
        self.assertEqual(text_out, "Searching." + SearchingEngine.ANSWER)
        delta = next(e for e in evs if e["type"] == "message_delta")
        self.assertEqual(delta["delta"]["stop_reason"], "end_turn")
        self.assertEqual(delta["usage"]["server_tool_use"], {"web_search_requests": 1})
        self.assertEqual(self.searx.queries, [("strata engine", "json")])
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertIn('"name": "web_search"', self.engine.prompts[0])
        self.assertIn('"query"', self.engine.prompts[0])
        second = self.engine.prompts[1]
        self.assertIn("[1] Strata engine\nURL: https://example.org/strata\nDate: 2026-10-01T00:00:00\n"
                      "Runs big MoE models.", second)
        self.assertNotIn("spam.example.com", second)

    def test_claude_code_shape_forced_without_thinking(self):
        """Claude Code's WebSearch: only the server tool, tool_choice naming it, thinking disabled."""
        self.start()
        code, text = self.post(stream=False, tool_choice={"type": "tool", "name": "web_search"},
                               thinking={"type": "disabled"},
                               tools=[{**WEB_SEARCH, "allowed_domains": ["example.net"], "max_uses": 8}])
        self.assertEqual(code, 200, text)
        msg = json.loads(text)
        self.assertEqual([b["type"] for b in msg["content"]], ["server_tool_use", "web_search_tool_result", "text"])
        self.assertEqual(msg["content"][0]["input"], {"query": "strata engine"})
        self.assertEqual([r["url"] for r in msg["content"][1]["content"]], ["https://docs.example.net/guide"])
        self.assertEqual(msg["content"][2]["text"], SearchingEngine.ANSWER)
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(self.searx.queries, [("strata engine site:example.net", "json")])
        self.assertTrue(self.engine.prompts[0].endswith("<function=web_search>\n"))
        self.assertFalse(self.engine.prompts[1].endswith("<function=web_search>\n"))

    def test_max_uses(self):
        self.start(searches=3)
        code, text = self.post(stream=False, tools=[{**WEB_SEARCH, "max_uses": 1}])
        self.assertEqual(code, 200, text)
        msg = json.loads(text)
        results = [b["content"] for b in msg["content"] if b["type"] == "web_search_tool_result"]
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(results[1], {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"})
        self.assertEqual(len(self.searx.queries), 1)
        self.assertEqual(msg["usage"]["server_tool_use"], {"web_search_requests": 1})
        self.assertIn("the search limit of this request (1) is reached", self.engine.prompts[2])

    def test_unreachable_searxng_is_an_error_result(self):
        self.start()
        self.svc.web_search = {**self.svc.web_search, "searxng": "http://127.0.0.1:9", "timeout_s": 2.0}
        code, text = self.post(stream=False)
        self.assertEqual(code, 200, text)
        msg = json.loads(text)
        result = next(b for b in msg["content"] if b["type"] == "web_search_tool_result")
        self.assertEqual(result["content"], {"type": "web_search_tool_result_error", "error_code": "unavailable"})
        self.assertIn("error: the search failed", self.engine.prompts[1])

    def test_without_searxng_the_client_gets_the_call(self):
        self.start(web_search=False)
        code, text = self.post(stream=False)
        self.assertEqual(code, 200, text)
        msg = json.loads(text)
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["content"][-1]["type"], "tool_use")
        self.assertEqual(msg["content"][-1]["input"], {"query": "strata engine"})
        self.assertEqual(self.searx.queries, [])


class History(unittest.TestCase):
    def test_server_blocks_become_a_call_and_its_result(self):
        result = websearch.block_content({"ok": True, "results": [
            {"url": "https://example.org", "title": "Example", "content": "snippet", "page_age": None}]})
        messages, tools, _ = anthropic_to_messages({"tools": [WEB_SEARCH], "messages": [
            {"role": "user", "content": "what is it?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Searching."},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "it"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": result},
                {"type": "text", "text": "It is an example."}]},
            {"role": "user", "content": "thanks"}]})
        self.assertEqual(messages[1:], [
            {"role": "assistant", "content": "Searching.",
             "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "it"}}}]},
            {"role": "tool", "content": "[1] Example\nURL: https://example.org\nsnippet"},
            {"role": "assistant", "content": "It is an example."},
            {"role": "user", "content": "thanks"}])
        self.assertEqual(tools, [{"name": "web_search", "description": websearch.DESCRIPTION,
                                  "parameters": websearch.SCHEMA}])

    def test_an_error_result(self):
        messages, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "q"}},
                {"type": "web_search_tool_result", "tool_use_id": "s",
                 "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}]}]})
        self.assertEqual(messages[-1], {"role": "tool", "content": "error: max_uses_exceeded"})


class Config(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(websearch.settings_from({}))
        s = websearch.settings_from({"web_search": {"searxng": "http://h:1/", "timeout_s": 5}})
        self.assertEqual(s, {"searxng": "http://h:1", "max_results": 10, "timeout_s": 5.0, "max_uses": 8})
        for bad in ({"searxng": ""}, {}, "http://h", {"searxng": "http://h", "max_results": 0},
                    {"searxng": "http://h", "max_uses": 1.5}, {"searxng": "http://h", "timeout_s": -1}):
            with self.subTest(block=bad), self.assertRaises(SystemExit):
                websearch.settings_from({"web_search": bad})


if __name__ == "__main__":
    unittest.main()
