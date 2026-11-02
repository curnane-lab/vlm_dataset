#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$PIPELINE_DIR${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'USAGE'
Usage:
  bash check_format.sh unit
  bash check_format.sh code
  bash check_format.sh formulas

Commands:
  unit      Run Python unittest checks for formula display rules.
  code      Compile all retained Python modules and syntax-check shell scripts.
  formulas  Run quick regression cases for formula normalization.
USAGE
}

cmd=${1:-}
case "$cmd" in
  unit)
    python3 -m unittest discover -s "$PIPELINE_DIR" -p 'test_*.py' -v
    ;;
  code)
    python3 -m py_compile \
      "$PIPELINE_DIR"/*.py
    for script in "$PIPELINE_DIR"/*.sh; do
      bash -n "$script"
    done
    echo "all pipeline code checks passed"
    ;;
  formulas)
    python3 - <<'PY'
import os
import sys
pipeline = "/data/verl_project/pipeline"
sys.path.insert(0, pipeline)
from formula_display import equation_display_ok, normalize_math

cases = [
    (r"\begin{equation}x=1\end{equation}", "$$\nx=1\n$$"),
    (r"\begin{align}x &= 1 \\ y &= 2\end{align}", "$$\n\\begin{aligned}x &= 1 \\\\ y &= 2\\end{aligned}\n$$"),
    (r"\begin{gather}a \\ b\end{gather}", "$$\n\\begin{gathered}a \\\\ b\\end{gathered}\n$$"),
    (r"\begin{array}{cc}1&2\\3&4\end{array}", "$$\n\\begin{array}{cc}1&2\\\\3&4\\end{array}\n$$"),
    (r"\mathbf{R}=\left[\begin{array}{cc}a&b\\c&d\end{array}\right]", None),
]

for source, expected in cases:
    result = normalize_math(source)
    assert result.action == "rewritten", result.as_dict()
    if expected is not None:
        assert result.text == expected, (source, result.text)
    assert equation_display_ok(result.text), result.as_dict()

assert normalize_math(r"\begin{align}x &= 1").reason == "unbalanced_environment"
assert normalize_math("$$x&=1\\\\y=2$$").reason == "stripped_alignment"
assert normalize_math("}x{").reason == "unbalanced_braces"
print("formula regression cases passed")
PY
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
