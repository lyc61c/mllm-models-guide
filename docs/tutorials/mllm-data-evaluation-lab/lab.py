"""Standard-library teaching lab. Synthetic metadata/predictions; no model run."""
import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path


def norm(s):
    """Toy normalization, NOT an official benchmark answer normalizer."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def anls(pred, refs):
    if not refs:
        raise ValueError("At least one reference is required")
    p = norm(pred)
    def score(ref):
        r = norm(ref)
        n = max(len(p), len(r))
        d = edit_distance(p, r) / n if n else 0.0
        return 1 - d if d < 0.5 else 0.0
    return max(score(r) for r in refs)


def vqa_consensus(pred, normalized_answers):
    """Leave-one-annotator-out core; inputs must ALREADY be normalized."""
    if len(normalized_answers) != 10:
        raise ValueError("This teaching example expects ten annotators")
    return sum(min(sum(a == pred for j, a in enumerate(normalized_answers)
                       if i != j) / 3, 1) for i in range(10)) / 10


def iou(a, b):
    """xyxy boxes in the same coordinate system; continuous area convention."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Expected xyxy boxes")
    if not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError("Non-finite coordinate")
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("Boxes must have positive area")
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area - overlap)


def group_components(rows):
    """Connect same-parent OR identical-media records BEFORE splitting."""
    parent = list(range(len(rows)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        x, y = find(i), find(j)
        parent[max(x, y)] = min(x, y)
    seen = {}
    for i, row in enumerate(rows):
        for key in (("parent", row["parent_id"]), ("media", row["media_sha256"])):
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def prepare(rows, protected_hashes, seed=17):
    groups = group_components(rows)
    kept, purged = [], []
    for group in groups:
        target = purged if any(r["media_sha256"] in protected_hashes for r in group) else kept
        target.append(group)
    # Toy exact dedup: identical media + identical normalized supervision.
    # Keep different questions on one media, but keep the whole group in one split.
    cleaned, removed = [], []
    for group in kept:
        unique, seen = [], set()
        for row in sorted(group, key=lambda r: r["id"]):
            key = (row["media_sha256"], norm(row["question"]), norm(row["answer"]))
            if key in seen:
                removed.append(row["id"])
            else:
                unique.append(row)
                seen.add(key)
        cleaned.append(unique)
    cleaned.sort(key=lambda g: min(r["id"] for r in g))
    random.Random(seed).shuffle(cleaned)
    n = len(cleaned)
    cuts = (int(n * 0.6), int(n * 0.8))
    split_groups = {"train": cleaned[:cuts[0]], "dev": cleaned[cuts[0]:cuts[1]],
                    "test": cleaned[cuts[1]:]}
    splits = {k: [r for g in v for r in g] for k, v in split_groups.items()}
    for field in ("parent_id", "media_sha256"):
        sets = {k: {r[field] for r in v} for k, v in splits.items()}
        assert not (sets["train"] & sets["dev"] or sets["train"] & sets["test"]
                    or sets["dev"] & sets["test"]), f"Leakage in {field}"
    return splits, {"seed": seed, "component_count_before_purge": len(groups),
                    "purged_ids": [r["id"] for g in purged for r in g],
                    "exact_duplicates_removed": removed,
                    "split_group_counts": {k: len(v) for k, v in split_groups.items()}}


def paired_group_bootstrap(records, seed=23, repeats=2000):
    """Equal-parent estimand; paired differences; empirical percentile interval."""
    groups = defaultdict(list)
    for r in records:
        groups[r["parent_id"]].append(r["score_b"] - r["score_a"])
    means = [sum(v) / len(v) for _, v in sorted(groups.items())]
    if len(means) < 2:
        raise ValueError("Need at least two independent parent groups")
    if repeats < 2:
        raise ValueError("Need at least two bootstrap replicates")
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(means) for _ in means) / len(means)
                   for _ in range(repeats))
    def quantile(p):
        pos = (len(draws) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(draws) - 1)
        return draws[lo] + (draws[hi] - draws[lo]) * (pos - lo)
    return {"estimand": "equal-parent mean paired score difference B-A",
            "delta": sum(means) / len(means), "percentile_95_interval": [quantile(.025), quantile(.975)],
            "independent_parent_groups": len(means), "repeats": repeats, "seed": seed}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("lab-output"))
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent
    rows = load_jsonl(fixture / "toy-records.jsonl")
    protected = set(json.loads((fixture / "protected-media.json").read_text()))
    splits, audit = prepare(rows, protected)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        (args.out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    predictions = load_jsonl(fixture / "toy-predictions.jsonl")
    scored = []
    for row in predictions:
        metric = anls if row["task"] == "ocr" else lambda p, refs: float(norm(p) in {norm(r) for r in refs})
        scored.append({"id": row["id"], "parent_id": row["parent_id"], "task": row["task"],
                       "score_a": metric(row["prediction_a"], row["references"]),
                       "score_b": metric(row["prediction_b"], row["references"])})
    report = {"warning": "SYNTHETIC TEACHING FIXTURE: no model, no real benchmark; predictions are separate from toy data splits",
              "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(fixture.glob("*.json*"))},
              "audit": audit, "split_record_counts": {k: len(v) for k, v in splits.items()},
              "paired_comparison": paired_group_bootstrap(scored), "per_example": scored}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_example"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
