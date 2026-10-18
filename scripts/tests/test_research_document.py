"""Research checks cover equations in diagrams and navigable theorem references."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "docs/research/occupancy/experiments/check_document.cjs"
FIXTURE = """# Research

## 1. 公式与证明

### 1.1 定理 1：测试条件

**定理陈述。** A declared claim.

**证明。** The derivation follows its statement.
"""


class ResearchDocumentTests(unittest.TestCase):
    def check_collection(self, files, entry="index.md"):
        script = """const {checkCollection} = require(process.argv[1]);
console.log(JSON.stringify(checkCollection(process.argv[2], process.argv[3])));"""
        with tempfile.TemporaryDirectory() as directory:
            for name, text in files.items():
                (Path(directory) / name).write_text(text)
            result = subprocess.run(
                ["node", "-e", script, str(CHECKER), directory, entry],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            return json.loads(result.stdout)

    def check_text(self, text):
        script = """const fs = require('fs');
const path = require('path');
const {checkDocument} = require(process.argv[1]);
const raw = fs.readFileSync(0, 'utf8');
console.log(JSON.stringify(checkDocument(raw, path.join(process.cwd(), 'docs/research/occupancy'))));"""
        result = subprocess.run(
            ["node", "-e", script, str(CHECKER)],
            cwd=ROOT,
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_valid_formula_inside_mermaid_is_counted(self):
        result = self.check_text(
            FIXTURE + '\n```mermaid\nflowchart TD\nA["$$\\frac{q}{p}$$"]\n```\n'
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["mermaid_diagrams"], 1)
        self.assertEqual(result["mermaid_math_expressions"], 1)
        self.assertEqual(result["math_expressions"], 1)

    def test_invalid_diagram_formula_cannot_hide_in_code_fence(self):
        result = self.check_text(
            FIXTURE
            + '\n```mermaid\nflowchart TD\nA["$$\\undefinedformula{x}$$"]\n```\n'
        )
        self.assertTrue(
            any("Undefined control sequence" in issue for issue in result["issues"])
        )

    def test_proof_must_follow_statement(self):
        result = self.check_text(
            FIXTURE.replace("**定理陈述。**", "**证明。**", 1).replace(
                "**证明。** The derivation", "**定理陈述。** The derivation"
            )
        )
        self.assertTrue(
            any("statement must precede proof" in issue for issue in result["issues"])
        )

    def test_missing_reference_and_number_gap_are_rejected(self):
        result = self.check_text(
            FIXTURE.replace("定理 1：", "定理 2：") + "\n引用定理 3。\n"
        )
        self.assertTrue(any("consecutive" in issue for issue in result["issues"]))
        self.assertTrue(
            any(
                "unknown theorem reference: 定理 3" in issue
                for issue in result["issues"]
            )
        )

    def test_chinese_toc_anchor_and_broken_anchor(self):
        valid = self.check_text(FIXTURE + "\n[正文](#1-公式与证明)\n")
        self.assertEqual(valid["issues"], [])
        invalid = self.check_text(FIXTURE + "\n[旧标题](#1-不存在)\n")
        self.assertIn("missing anchor: #1-不存在", invalid["issues"])

    def test_unavailable_external_source_is_not_an_internal_break(self):
        result = self.check_text(
            FIXTURE
            + "\n[论文](../../../../Awesome-Paper/nonexistent-research-fixture.tex)\n"
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(len(result["unavailable_external_links"]), 1)

    def test_guide_and_independently_numbered_pages_are_valid(self):
        result = self.check_collection(
            {
                "index.md": "# Guide\n\n[A](a.md) [B](b.md)\n",
                "a.md": FIXTURE + "\n[B·定理 1](b.md#11-定理-1测试条件)\n",
                "b.md": FIXTURE,
            }
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["document_count"], 3)
        self.assertEqual(result["theorem_statements"], 2)

    def test_cross_page_anchor_must_exist(self):
        result = self.check_collection(
            {
                "index.md": "# Guide\n\n[A](a.md)\n",
                "a.md": FIXTURE + "\n[B·定理 1](b.md#missing-theorem)\n",
                "b.md": FIXTURE,
            }
        )
        self.assertTrue(
            any(
                "missing linked anchor: b.md#missing-theorem" in issue
                for issue in result["issues"]
            )
        )

    def test_unlinked_topic_cannot_be_silently_omitted(self):
        result = self.check_collection(
            {
                "index.md": "# Guide\n\n[A](a.md)\n",
                "a.md": FIXTURE,
                "orphan.md": FIXTURE,
            }
        )
        self.assertIn("document is unreachable from entry: orphan.md", result["issues"])

    def test_empty_collection_and_missing_entry_are_failures(self):
        empty = self.check_collection({})
        self.assertIn("no research documents selected", empty["issues"])
        missing = self.check_collection({"a.md": FIXTURE})
        self.assertIn("missing entry document: index.md", missing["issues"])

    def test_formula_in_a_later_topic_is_checked(self):
        result = self.check_collection(
            {
                "index.md": "# Guide\n\n[A](a.md)\n",
                "a.md": FIXTURE + "\n[B](b.md)\n",
                "b.md": FIXTURE + "\n$$\\undefinedformula{x}$$\n",
            }
        )
        self.assertTrue(
            any(
                issue.startswith("b.md:") and "Undefined control sequence" in issue
                for issue in result["issues"]
            )
        )

    def test_footnote_must_be_defined_on_its_own_page(self):
        result = self.check_collection(
            {
                "index.md": "# Guide\n\n[A](a.md) [B](b.md)\n",
                "a.md": FIXTURE + "\nCitation[^source].\n",
                "b.md": FIXTURE + "\n[^source]: Source only defined here.\n",
            }
        )
        self.assertIn("a.md: missing footnote: source", result["issues"])


if __name__ == "__main__":
    unittest.main()
