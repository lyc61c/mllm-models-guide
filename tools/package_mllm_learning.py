"""Package only the reviewed data/evaluation tutorial and teaching fixtures."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/tutorials"
PREFIX = "mllm-data-evaluation"


def main():
    source = DOC / (PREFIX + ".md")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    view = DOC / (PREFIX + ".html")
    view_sha = hashlib.sha256(view.read_bytes()).hexdigest()
    review = json.loads((DOC / (PREFIX + ".html.review.json")).read_text(encoding="utf-8"))
    assert review["verdict"] == "PASS"
    assert review["source_sha256"] == source_sha and review["html_sha256"] == view_sha
    checks = json.loads((DOC / (PREFIX + ".checks.json")).read_text(encoding="utf-8"))
    assert checks["status"] == "PASS" and checks["source_sha256"] == source_sha
    names = [PREFIX + x for x in (".md", ".html", ".sources.json", ".render.json", ".checks.json", ".html.review.json", "-READING.md")]
    paths = [DOC / x for x in names]
    paths += sorted((DOC / "assets/data-evaluation").glob("*.svg"))
    paths += sorted(p for p in (DOC / (PREFIX + "-lab")).iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl"))
    entries = [{"path": p.relative_to(DOC).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in paths]
    manifest = {"article": PREFIX, "source_sha256": source_sha, "html_sha256": view_sha,
                "review_verdict": review["verdict"], "files": entries,
                "note": "Only this tutorial; toy data/predictions, no model evaluation. The separate model guide is not bundled."}
    output = DOC / (PREFIX + ".zip")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p, entry in zip(paths, entries):
            archive.writestr(PREFIX + "/" + entry["path"], p.read_bytes())
        archive.writestr(PREFIX + "/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for entry in entries:
            assert hashlib.sha256(archive.read(PREFIX + "/" + entry["path"])).hexdigest() == entry["sha256"]
    output.with_suffix(".package.json").write_text(json.dumps({**manifest, "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "zip_bytes": output.stat().st_size}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(entries) + 1, "zip_bytes": output.stat().st_size, "source_sha256": source_sha}))


if __name__ == "__main__":
    main()
