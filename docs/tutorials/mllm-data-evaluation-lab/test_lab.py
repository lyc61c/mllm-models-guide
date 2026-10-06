import unittest
from lab import anls, edit_distance, group_components, iou, paired_group_bootstrap, prepare, vqa_consensus


class LabTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(anls("ABCD", ["abce"]), .75)
        self.assertEqual(anls("ab", ["ac"]), 0)  # strict < 0.5
        self.assertEqual(anls("", [""]), 1)  # explicitly toy-defined behavior
        self.assertEqual(anls("bad", ["bad", "other"]), 1)
        for count, expected in [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)]:
            self.assertAlmostEqual(vqa_consensus("yes", ["yes"] * count + ["no"] * (10-count)), expected)
        self.assertAlmostEqual(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1/7)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)
        with self.assertRaises(ValueError):
            iou([0, 0, 0, 1], [0, 0, 1, 1])

    def test_transitive_purge_and_dedup(self):
        def row(i, parent, media):
            return dict(id=i, parent_id=parent, media_sha256=media, question="Q", answer="A")
        rows = [row("1", "p1", "a"), row("2", "p1", "b"), row("3", "p2", "b"),
                row("4", "p2", "c"), row("5", "p3", "d"), row("6", "p4", "d")]
        self.assertEqual(sorted(map(len, group_components(rows))), [2, 4])
        splits, audit = prepare(rows, {"a"})
        self.assertEqual(audit["purged_ids"], ["1", "2", "3", "4"])
        self.assertEqual(audit["exact_duplicates_removed"], ["6"])
        self.assertEqual(sum(map(len, splits.values())), 1)

    def test_bootstrap_is_paired_group_balanced_and_reproducible(self):
        rows = [dict(parent_id="p1", score_a=0, score_b=1)] * 10 + [dict(parent_id="p2", score_a=1, score_b=0)]
        a = paired_group_bootstrap(rows, repeats=100)
        self.assertEqual(a["delta"], 0)  # equal-parent, NOT 9/11 question-weighted
        self.assertEqual(a, paired_group_bootstrap(rows, repeats=100))
        with self.assertRaises(ValueError):
            paired_group_bootstrap(rows[:1])


if __name__ == "__main__":
    unittest.main()
