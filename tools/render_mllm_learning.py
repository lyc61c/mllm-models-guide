"""Generate the standalone coursebook page from its canonical Markdown."""
import base64
import hashlib
import html
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/tutorials"
SOURCE = DOC / "mllm-data-evaluation.md"
OUTPUT = SOURCE.with_suffix(".html")
MD = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")


def render():
    text = SOURCE.read_text(encoding="utf-8")
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    math = {}
    def protect(m):
        key = f"LEARNINGMATH{len(math)}PLACEHOLDER"
        value = m.group()
        cls = "display" if value.startswith(("$$", "\\[")) else "inline"
        math[key] = f'<span class="math {cls}">{html.escape(value)}</span>'
        return key
    pattern = r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\\)\$(?!\$)[^\n$]+?(?<!\\)\$'
    parts = re.split(r'(^```.*?^```[^\n]*$)', text, flags=re.M | re.S)
    text = "".join(p if p.startswith("```") else re.sub(pattern, protect, p) for p in parts)
    tokens = MD.parse(text)
    # Only author-controlled details/summary markup is permitted in source HTML.
    for token in tokens:
        raw = [token] if token.type in ("html_block", "html_inline") else []
        raw += [c for c in (token.children or []) if c.type == "html_inline"]
        for item in raw:
            rest = re.sub(r'</?(?:details|summary)>', '', item.content).strip()
            if rest and item.type != "html_inline":
                # A Markdown HTML block can contain summary text on the same line.
                rest = re.sub(r'<summary>[^<>]*</summary>', '', item.content)
                rest = re.sub(r'</?details>', '', rest).strip()
            if rest:
                raise ValueError("Unexpected raw HTML in source: " + item.content[:120])
    toc = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            label = tokens[i + 1].content
            hid = "section-" + str(len(toc) + 1)
            token.attrSet("id", hid)
            toc.append((int(token.tag[1:]), label, hid))
        for child in token.children or []:
            if child.type == "link_open":
                url = child.attrGet("href") or ""
                if not (url.startswith(("https://", "#")) or re.fullmatch(r'[\w./-]+', url)):
                    raise ValueError("Unsafe link: " + url)
    body = MD.renderer.render(tokens, MD.options, {})
    for key, value in math.items():
        body = body.replace(key, value)
    images = []
    def embed(m):
        src = html.unescape(m.group(1))
        path = (SOURCE.parent / src).resolve()
        assert path.is_relative_to(SOURCE.parent.resolve()) and path.suffix == ".svg" and path.is_file()
        raw = path.read_bytes()
        assert not re.search(rb'<(?:script|foreignObject)|\bon\w+\s*=|(?:href|src)\s*=', raw, re.I)
        images.append({"path": src, "sha256": hashlib.sha256(raw).hexdigest()})
        return 'src="data:image/svg+xml;base64,' + base64.b64encode(raw).decode() + '"'
    body = re.sub(r'src="([^"]+)"', embed, body)
    nav = ''.join(f'<a class="level-{level}" href="#{hid}">{html.escape(label)}</a>'
                  for level, label, hid in toc if level in (2, 3))
    title = toc[0][1]
    css = '''
    :root{--ink:#182638;--muted:#5a687b;--blue:#2255b4;--line:#dde5ef;--soft:#f2f6fc}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:76px}body{margin:0;color:var(--ink);background:#fff;font:16px/1.95 system-ui,"Microsoft YaHei","Noto Sans CJK SC",sans-serif}a{color:var(--blue);text-underline-offset:3px;overflow-wrap:anywhere}
    .topbar{position:sticky;top:0;z-index:10;background:#ffffffed;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:12px 28px;display:flex;gap:24px;align-items:center;font-size:14px}.brand{font-weight:750;margin-right:auto}.topbar a{text-decoration:none}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:48px;max-width:1380px;margin:auto;padding:36px 32px 80px}
    nav{position:sticky;top:85px;align-self:start;max-height:calc(100vh - 110px);overflow:auto;font-size:12px;padding-right:20px;border-right:1px solid var(--line)}nav summary{font-size:14px;font-weight:700;cursor:pointer;margin-bottom:12px}nav a{display:block;text-decoration:none;line-height:1.6;padding:5px 4px;color:var(--muted)}nav a:hover{color:var(--blue);background:var(--soft)}nav .level-2{font-size:13px;font-weight:700;color:var(--ink);margin-top:16px}nav .level-3{padding-left:13px}
    main{min-width:0;max-width:990px}.meta{padding:12px 16px;border:1px solid var(--line);border-radius:8px;background:var(--soft);font-size:11px;color:var(--muted);line-height:1.75;overflow-wrap:anywhere}.eyebrow{font-size:11px;font-weight:750;color:var(--blue);letter-spacing:.12em;display:block;margin-bottom:7px}h1{font-size:36px;line-height:1.45;letter-spacing:-.02em;margin:28px 0}h2{font-size:28px;line-height:1.5;margin:64px 0 22px;padding:20px 0 10px;border-top:3px solid var(--blue)}h3{font-size:22px;line-height:1.5;margin:42px 0 18px}p{margin:16px 0}li{margin:9px 0}strong{font-weight:720}
    blockquote{margin:24px 0;border-left:4px solid #7c9cd6;padding:8px 20px;background:var(--soft);font-size:15px}blockquote p{margin:8px 0}img{display:block;width:100%;height:auto;margin:26px 0 10px;border:1px solid var(--line);border-radius:8px}p:has(>img)+p:has(>em){font-size:12px;color:var(--muted);line-height:1.8;margin-top:8px}
    table{display:block;max-width:100%;overflow-x:auto;border-collapse:collapse;font-size:13px;line-height:1.8;margin:24px 0}th,td{padding:11px 13px;min-width:120px;vertical-align:top;text-align:left;border:1px solid var(--line)}th{background:#eaf1fc;font-weight:700}tbody tr:nth-child(even){background:#f8fafc}pre{overflow-x:auto;border-radius:8px;background:#152238;color:#e6edf7;padding:20px;font:13px/1.75 Consolas,"Cascadia Code",monospace}code{font-family:Consolas,"Cascadia Code",monospace;font-size:.88em;background:#eef2f7;padding:2px 5px;border-radius:3px}pre code{font-size:inherit;background:none;padding:0}
    details{border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:18px 0}summary{cursor:pointer;font-weight:650}nav details{border:0;padding:0;margin:0}.math.display{display:block;max-width:100%;overflow-x:auto;padding:16px 0}mjx-container[jax="CHTML"][display="true"]{margin:0!important}footer{border-top:1px solid var(--line);margin-top:45px;padding-top:20px;font-size:12px;color:var(--muted);overflow-wrap:anywhere}
    @media(max-width:1050px){.layout{grid-template-columns:215px minmax(0,1fr);gap:25px;padding:28px 24px}h1{font-size:31px}}
    @media(max-width:760px){.topbar{padding:10px 16px;gap:12px;font-size:12px}.layout{display:block;padding:20px 18px 60px}nav{position:relative;top:0;max-height:250px;border-right:0;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:24px}h1{font-size:28px}h2{font-size:25px}h3{font-size:21px}body{font-size:16px}table{font-size:12px}th,td{padding:9px;min-width:110px}.brand{max-width:42%;line-height:1.4}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
    @media print{.topbar,nav{display:none}.layout{display:block;padding:0}main{max-width:none}body{font-size:11pt}h2{margin-top:30px}h2,h3{break-after:avoid}img,pre{break-inside:avoid}pre{white-space:pre-wrap;color:#182638;background:#f1f5f9}table{overflow:visible;font-size:9pt}a{color:inherit}}
    '''
    part1 = next(hid for level, label, hid in toc if label.startswith("第一部分"))
    part2 = next(hid for level, label, hid in toc if label.startswith("第二部分"))
    result = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="中文 MLLM 学习教程：多模态数据获取、清洗、对齐、去重、数据泄漏、评估协议、基准指标与可运行练习。"><meta name="source-path" content="{html.escape(str(SOURCE))}"><meta name="source-sha256" content="{sha}"><meta name="generated-at" content="{now}"><title>{html.escape(title)}</title><style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script></head>
<body><header class="topbar"><a class="brand" href="mllm-expanded.html">MLLM 学习资料</a><a href="#{part1}">数据工程</a><a href="#{part2}">评估基准</a><a href="mllm-data-evaluation.zip">下载学习包</a></header><div class="layout"><nav aria-label="教程目录"><details open><summary>本页目录</summary>{nav}</details></nav><main><div class="meta"><span class="eyebrow">MULTIMODAL LEARNING · DATA &amp; EVALUATION</span>资料核验：2026-10-06 · 生成：{now}<br>可编辑源文件：{html.escape(str(SOURCE))}<br>源文件 SHA256：{sha}</div>{body}<footer>正文由 Markdown 生成；5 张原创图示内嵌。公式使用 MathJax 在线排版，离线时保留 TeX。<br><a href="mllm-data-evaluation.md">Markdown</a> · <a href="mllm-data-evaluation.sources.json">来源登记</a> · <a href="mllm-data-evaluation.html.review.json">审阅记录</a><br>源文件 SHA256：{sha}</footer></main></div></body></html>'''
    assert not re.search(r'LEARNINGMATH\d+PLACEHOLDER', result)
    OUTPUT.write_text(result, encoding="utf-8")
    report = {"source_path": str(SOURCE), "source_sha256": sha, "generated_at": now,
              "headings": toc, "images": images, "math_expressions": len(math),
              "html_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              "html_bytes": OUTPUT.stat().st_size,
              "external_runtime": "MathJax 3.2.2 via jsDelivr; offline TeX remains readable"}
    OUTPUT.with_suffix(".render.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("headings", "images")}, ensure_ascii=False))


if __name__ == "__main__":
    render()
