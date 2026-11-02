#!/usr/bin/env python3
"""Regression checks for formula display normalization.

These lock the samples that were compiled or rejected in review:
inline delimiters, amsmath shells, preserved body text, and the audit
versus trainable export policy. pdflatex is not required.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from formula_display import equation_display_ok, normalize_math  # noqa: E402


def norm(text):
    return normalize_math(text)


class FormulaDisplayTest(unittest.TestCase):
    def test_inline_paren_becomes_display(self):
        result = norm(r"\(x=1\)")
        self.assertEqual(result.action, "rewritten")
        self.assertEqual(result.change_type, "strip_inline")
        self.assertEqual(result.text, "$$\nx=1\n$$")
        self.assertNotIn(r"\(", result.text)
        self.assertTrue(equation_display_ok(result.text))
        self.assertFalse(equation_display_ok(r"\(x=1\)"))
        self.assertFalse(equation_display_ok(r"$$ \(x=1\) $$"))

    def test_inline_dollar_becomes_display(self):
        result = norm("$x=1$")
        self.assertEqual(result.text, "$$\nx=1\n$$")
        self.assertEqual(result.change_type, "strip_inline")
        self.assertNotIn("$x", result.text)
        self.assertTrue(equation_display_ok(result.text))
        self.assertFalse(equation_display_ok("$x=1$"))
        self.assertFalse(equation_display_ok("$$ $x=1$ $$"))

    def test_escaped_dollar_is_not_a_delimiter(self):
        result = norm(r"\$x=1")
        self.assertEqual(result.change_type, "wrap_plain")
        self.assertIn(r"\$x=1", result.text)
        self.assertTrue(equation_display_ok(result.text))

    def test_mismatched_and_residual_delimiters_review(self):
        mismatched = norm(r"\(x=1$")
        self.assertEqual(mismatched.action, "review")
        self.assertEqual(mismatched.reason, "mismatched_delimiter")
        residual = norm(r"x $y$ z")
        self.assertEqual(residual.action, "review")
        self.assertEqual(residual.reason, "residual_delimiter")
        self.assertFalse(equation_display_ok(r"$$ x $y$ z $$"))

    def test_starred_shells_are_not_amsmath(self):
        for src in (
                r"\begin{array*}{c} a \end{array*}",
                r"\begin{aligned*} a &= b \end{aligned*}",
                r"\begin{gathered*} a \\ b \end{gathered*}"):
            result = norm(src)
            self.assertEqual(result.action, "review", src)
            self.assertEqual(result.reason, "unknown_structure", src)
            self.assertFalse(equation_display_ok("$$\n" + src + "\n$$"))

    def test_align_and_gather_convert_without_dropping_body(self):
        align = norm(r"\begin{align} a &= b \end{align}")
        self.assertEqual(
            align.text, "$$\n\\begin{aligned} a &= b \\end{aligned}\n$$")
        self.assertIn("a &= b", align.text)
        self.assertTrue(equation_display_ok(align.text))
        gather = norm(r"\begin{gather} a \\ b \end{gather}")
        self.assertEqual(
            gather.text, "$$\n\\begin{gathered} a \\\\ b \\end{gathered}\n$$")
        self.assertTrue(equation_display_ok(gather.text))
        starred = norm(r"\begin{align*}{x}+y\end{align*}")
        self.assertIn("{x}+y", starred.text)
        self.assertIn(r"\begin{aligned}", starred.text)
        self.assertNotIn("align*", starred.text)

    def test_array_and_aligned_keep_shell_and_arguments(self):
        array = norm(r"\begin{array}{cc} a & b \\ c & d \end{array}")
        self.assertIn(r"\begin{array}{cc}", array.text)
        self.assertIn(r"\end{array}", array.text)
        self.assertIn(r"a & b", array.text)
        self.assertTrue(equation_display_ok(array.text))
        aligned = norm(r"\begin{aligned} a &= b \end{aligned}")
        self.assertEqual(
            aligned.text, "$$\n\\begin{aligned} a &= b \\end{aligned}\n$$")
        self.assertTrue(equation_display_ok(aligned.text))

    def test_equation_keeps_braces_and_brackets(self):
        braced = norm(r"\begin{equation}{x}+y\end{equation}")
        self.assertEqual(braced.text, "$$\n{x}+y\n$$")
        bracket = norm(r"\begin{equation}[a,b]\end{equation}")
        self.assertEqual(bracket.text, "$$\n[a,b]\n$$")
        self.assertTrue(equation_display_ok(braced.text))
        self.assertTrue(equation_display_ok(bracket.text))

    def test_stripped_alignment_and_broken_environments_review(self):
        stripped = norm(r"$$x&=1\\y&=2$$")
        self.assertEqual(stripped.reason, "stripped_alignment")
        self.assertFalse(equation_display_ok(r"$$x&=1\\y&=2$$"))
        self.assertEqual(norm(r"\end{array}").reason, "unbalanced_environment")
        self.assertEqual(
            norm(r"\begin{cases} a").reason, "unbalanced_environment")
        self.assertEqual(
            norm(r"\begin{cases} a \end{cases}").reason, "unknown_structure")

    def test_repeat_is_idempotent(self):
        samples = [
            r"\(x=1\)",
            "$x=1$",
            r"\begin{align} a &= b \end{align}",
            r"\begin{array}{c} a \\ b \end{array}",
            r"\begin{equation}{x}+y\end{equation}",
            r"x^2",
        ]
        for src in samples:
            once = norm(src)
            twice = norm(once.text)
            self.assertEqual(twice.action, "unchanged", src)
            self.assertEqual(twice.text, once.text, src)

    def test_distill_uses_the_same_inline_rule(self):
        import types
        for name in ("pandas", "pyarrow", "pyarrow.parquet", "requests", "tqdm"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["tqdm"].tqdm = lambda *args, **kwargs: args[0] if args else None
        import distill_qa

        self.assertEqual(
            distill_qa.normalize_display_equation(r"\(x=1\)"), "$$\nx=1\n$$")
        self.assertEqual(
            distill_qa.normalize_display_equation(r"$$ \(x=1\) $$"),
            "$$\nx=1\n$$")
        self.assertFalse(distill_qa.equation_display_ok(r"$$ \(x=1\) $$"))
        self.assertFalse(distill_qa.equation_display_ok("$$ $x=1$ $$"))


class ExportPolicyTest(unittest.TestCase):
    def test_audit_keeps_review_rows_and_trainable_omits_them(self):
        import types
        for name in ("pyarrow", "pyarrow.parquet"):
            sys.modules.setdefault(name, types.ModuleType(name))
        import normalize_formula_delims as offline

        self.assertTrue(offline.include_in_output("review", "audit"))
        self.assertTrue(offline.include_in_output("rewritten", "trainable"))
        self.assertFalse(offline.include_in_output("review", "trainable"))

    def test_same_root_is_rejected(self):
        import types
        for name in ("pyarrow", "pyarrow.parquet"):
            sys.modules.setdefault(name, types.ModuleType(name))
        import normalize_formula_delims as offline

        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(offline.output_roots_collide(directory, directory))
            nested = os.path.join(directory, "out")
            os.makedirs(nested)
            self.assertFalse(offline.output_roots_collide(directory, nested))
            link = os.path.join(directory, "link")
            os.symlink(directory, link)
            self.assertTrue(offline.output_roots_collide(directory, link))


if __name__ == "__main__":
    unittest.main()
