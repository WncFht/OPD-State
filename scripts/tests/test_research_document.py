"""Research checks cover equations in diagrams and navigable theorem references."""

import json
import subprocess
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


if __name__ == "__main__":
    unittest.main()
