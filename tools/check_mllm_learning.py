"""Static integrity checks; no claim of browser/MathJax runtime validation."""
import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/tutorials"
SOURCE = DOC / "mllm-data-evaluation.md"


class Audit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.ids = []
        self.fragments = []
        self.images = []
        self.local_links = []
        self.in_code = False
        self.codes = []
        self.code = ""
    def handle_starttag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        a = dict(attrs)
        assert not any(k.lower().startswith("on") for k in a), "Event handler"
        assert tag not in ("iframe", "object", "embed", "form", "input", "base")
        if "id" in a:
            self.ids.append(a["id"])
        href = a.get("href", "")
        if href.startswith("#"):
            self.fragments.append(href[1:])
        elif href and not href.startswith("https://"):
            assert not re.match(r"[a-z]+:", href, re.I), "Unsafe URL scheme"
            self.local_links.append(href)
        if tag == "img":
            assert a.get("alt")
            assert a["src"].startswith("data:image/svg+xml;base64,")
            self.images.append(base64.b64decode(a["src"].split(",", 1)[1]))
        if tag == "code" and self.counts.get("pre", 0) > len(self.codes):
            self.in_code = True
            self.code = ""
    def handle_endtag(self, tag):
        if tag == "code" and self.in_code:
            self.codes.append(self.code)
            self.in_code = False
    def handle_data(self, text):
        if self.in_code:
            self.code += text


def main():
    raw = SOURCE.read_text(encoding="utf-8")
    content = SOURCE.with_suffix(".html").read_text(encoding="utf-8")
    manifest = json.loads(SOURCE.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest["source_sha256"]
    assert manifest["source_sha256"] in content
    assert hashlib.sha256(SOURCE.with_suffix(".html").read_bytes()).hexdigest() == manifest["html_sha256"]
    registry = json.loads(SOURCE.with_suffix(".sources.json").read_text(encoding="utf-8"))
    assert registry["source_sha256"] == manifest["source_sha256"]
    parser = Audit()
    parser.feed(content)
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate IDs"
    assert set(parser.fragments) <= set(parser.ids), "Broken TOC links"
    assert parser.counts.get("img") == len(manifest["images"]) == 5
    assert parser.counts.get("script") == 2  # trusted MathJax config + versioned CDN
    assert parser.counts.get("details") == 7 and parser.counts.get("summary") == 7
    tokens = MarkdownIt("commonmark", {"html": True}).enable("table").parse(raw)
    code = [t.content for t in tokens if t.type == "fence"]
    assert code == parser.codes, "Code block content changed"
    assert sum(t.type == "table_open" for t in tokens) == parser.counts["table"]
    assert sum(t.type == "heading_open" for t in tokens) == len(manifest["headings"])
    for blob, info in zip(parser.images, manifest["images"]):
        assert hashlib.sha256(blob).hexdigest() == info["sha256"]
        assert blob == (DOC / info["path"]).read_bytes()
    assert not re.search(r"LEARNINGMATH\d+PLACEHOLDER|\{\{(?:TITLE|BODY|TOC)\}\}", content)
    math_values = re.findall(r'<span class="math (?:inline|display)">([\s\S]*?)</span>', content)
    assert len(math_values) == manifest["math_expressions"]
    # The package/review sidecar is created after the frozen-artifact review.
    deferred = {"mllm-data-evaluation.zip", "mllm-data-evaluation.html.review.json"}
    assert all((DOC / link).is_file() or link in deferred for link in parser.local_links)
    result = {"status": "PASS", "scope": "static integrity; no browser runtime claim",
              "source_sha256": manifest["source_sha256"], "html_sha256": manifest["html_sha256"],
              "headings": len(manifest["headings"]), "tables": parser.counts["table"],
              "code_blocks": len(code), "math_expressions": len(math_values), "embedded_figures": len(parser.images),
              "exercise_details": 6, "registered_sources": len(registry["sources"])}
    SOURCE.with_suffix(".checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
