# -*- coding: utf-8 -*-
"""Shared display-formula normalization for distillation, offline repair, and QC.

Rules, applied only when the structure is unambiguous:

- aligned / array: keep the whole environment, including its argument
- align / align*: convert to aligned, body unchanged
- gather / gather*: convert to gathered, body unchanged
- equation / equation*: drop only that environment shell, body unchanged
- a formula wrapped entirely by one ``\\(...\\)`` or ``$...$`` pair: drop
  that pair, then normalize. Escaped ``\\$`` is not a delimiter. A leftover
  or mismatched delimiter is review
- anything else with an environment, a bare alignment ``&``, or an
  unbalanced begin/end: review, do not guess

``array*``, ``aligned*``, and ``gathered*`` are not amsmath environments
and are not accepted. A successful result is ``$$\\n...\\n$$``. Running it
again is a no-op. ``{x}+y`` and ``[a,b]`` are formula text, not arguments.
"""

import re

_ENV_TOKEN_RE = re.compile(r"\\(begin|end)\{([^{}]+)\}")
_SINGLE_ENV_RE = re.compile(
    r"^\s*\\begin\{(?P<env>[^{}]+)\}(?P<body>.*)\\end\{(?P=env)\}\s*$",
    re.DOTALL,
)
_DISPLAY_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]", re.DOTALL)
_ONE_DISPLAY_RE = re.compile(r"\$\$(.*)\$\$|\\\[(.*)\\\]", re.DOTALL)
_TOP_DISPLAY_RE = re.compile(
    r"\\begin\{(?:equation\*|align\*|gather\*|eqnarray\*"
    r"|equation|align|gather|eqnarray)\}"
)

# Starred array/aligned/gathered are not defined by amsmath.
_KEEP_SHELL = {"aligned", "array", "gathered"}
_CONVERT = {
    "align": "aligned",
    "align*": "aligned",
    "gather": "gathered",
    "gather*": "gathered",
}
_STRIP_SHELL = {"equation", "equation*"}


class FormulaNorm(object):
    """One normalization decision. ``text`` is the replacement only when
    ``action`` is ``unchanged`` or ``rewritten``. Review leaves the caller
    holding the original answer."""

    def __init__(self, text, action, change_type, reason):
        self.text = text
        self.action = action
        self.change_type = change_type
        self.reason = reason

    def as_dict(self):
        return {
            "text": self.text,
            "action": self.action,
            "change_type": self.change_type,
            "reason": self.reason,
        }


def _review(reason):
    return FormulaNorm("", "review", "review", reason)


def _wrap(body):
    return "$$\n" + body.strip() + "\n$$"


def env_balance_error(text):
    """Return ``unbalanced_environment`` when begin/end names do not pair.

    This is a stack check over every environment, not a count of the few
    names the rewriter knows. An isolated ``\\end{array}`` and a truncated
    ``cases`` both fail.
    """
    stack = []
    for kind, env in _ENV_TOKEN_RE.findall(text or ""):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            return "unbalanced_environment"
    if stack:
        return "unbalanced_environment"
    return None


def _escaped(text, index):
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 1


def has_bare_alignment(text):
    """True when ``&`` sits outside every environment.

    ``&`` inside aligned/array/gathered/cases is alignment syntax.
    ``&`` with no environment is a shell that was already stripped; the
    original environment cannot be recovered from it.
    """
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith(r"\begin{", index):
            depth += 1
            index += len(r"\begin{")
            continue
        if text.startswith(r"\end{", index):
            depth = max(0, depth - 1)
            index += len(r"\end{")
            continue
        if text[index] == "&" and depth == 0 and not _escaped(text, index):
            return True
        index += 1
    return False


def _brace_balanced(text):
    stripped = (text or "").replace(r"\{", "").replace(r"\}", "")
    return stripped.count("{") == stripped.count("}")


def has_top_display_env(text):
    """True when equation/align/gather/eqnarray is still present.

    aligned, array, and gathered are legal inside ``$$`` and do not match.
    """
    return _TOP_DISPLAY_RE.search(text or "") is not None


def _display_blocks(text):
    stripped = text.strip()
    matches = list(_DISPLAY_BLOCK_RE.finditer(stripped))
    if not matches:
        return None
    pos = 0
    bodies = []
    for matched in matches:
        if stripped[pos:matched.start()].strip():
            return None
        body = matched.group(1) if matched.group(1) is not None else matched.group(2)
        bodies.append((body or "").strip())
        pos = matched.end()
    if stripped[pos:].strip():
        return None
    return bodies


def _single_env(text):
    matched = _SINGLE_ENV_RE.match(text)
    if not matched:
        return None
    body = matched.group("body")
    if env_balance_error(body):
        return None
    return matched.group("env"), body


def _finish(original, wrapped, change_type):
    if wrapped.strip() == original.strip():
        return FormulaNorm(wrapped, "unchanged", "unchanged", "")
    return FormulaNorm(wrapped, "rewritten", change_type, "")


def normalize_math(answer):
    """Normalize one formula answer. See the module docstring for the rules."""
    original = answer if isinstance(answer, str) else ""
    if not original.strip():
        return _review("empty")
    balance = env_balance_error(original)
    if balance:
        return _review(balance)
    if not _brace_balanced(original):
        return _review("unbalanced_braces")
    return _normalize(original.strip())


def _scan_delimiters(text):
    """Pair unescaped ``$$``, ``\\[\\]``, ``$``, and ``\\(\\)``.

    Return ``(tokens, error)``. Each token is ``(kind, start, end)`` and
    covers the delimiters plus their interior. An odd run of backslashes
    escapes the following character, so ``\\$`` is not a dollar.
    """
    tokens = []
    index = 0
    limit = len(text)
    while index < limit:
        if text.startswith("$$", index) and not _escaped(text, index):
            close = _find_token(text, "$$", index + 2)
            if close is None:
                return None, "mismatched_delimiter"
            tokens.append(("display_dollar", index, close + 2))
            index = close + 2
            continue
        if text.startswith("\\[", index) and not _escaped(text, index):
            close = _find_token(text, "\\]", index + 2)
            if close is None:
                return None, "mismatched_delimiter"
            tokens.append(("display_bracket", index, close + 2))
            index = close + 2
            continue
        if index < limit and text[index] == "$" and not _escaped(text, index):
            close = _find_dollar(text, index + 1)
            if close is None:
                return None, "mismatched_delimiter"
            tokens.append(("inline_dollar", index, close + 1))
            index = close + 1
            continue
        if text.startswith("\\(", index) and not _escaped(text, index):
            close = _find_token(text, "\\)", index + 2)
            if close is None:
                return None, "mismatched_delimiter"
            tokens.append(("inline_paren", index, close + 2))
            index = close + 2
            continue
        if ((text.startswith("\\)", index) or text.startswith("\\]", index))
                and not _escaped(text, index)):
            return None, "mismatched_delimiter"
        index += 1
    return tokens, None


def _find_token(text, token, start):
    cursor = start
    while cursor <= len(text) - len(token):
        if text.startswith(token, cursor) and not _escaped(text, cursor):
            return cursor
        cursor += 1
    return None


def _find_dollar(text, start):
    cursor = start
    while cursor < len(text):
        if text[cursor] == "$" and not _escaped(text, cursor):
            return cursor
        cursor += 1
    return None


def _covering_inline(text):
    """Return ``(body, reason)`` when ``text`` is one inline pair or invalid.

    ``(None, None)`` means there is no inline delimiter to peel.
    """
    tokens, error = _scan_delimiters(text)
    if error:
        return None, error
    if not tokens:
        return None, None
    kind, start, end = tokens[0]
    if (len(tokens) == 1 and start == 0 and end == len(text)
            and kind in ("inline_dollar", "inline_paren")):
        skip = 1 if kind == "inline_dollar" else 2
        return text[start + skip:end - skip], None
    return None, "residual_delimiter"


def _normalize(text):
    blocks = _display_blocks(text)
    if blocks is not None and not (
            len(blocks) == 1 and _ONE_DISPLAY_RE.fullmatch(text)):
        return _normalize_blocks(text, blocks)

    peeled, did_peel = _peel_display(text)
    if did_peel:
        if not peeled:
            return _review("empty")
        inner = _normalize(peeled)
        if inner.action == "review":
            return inner
        return _finish(text, inner.text, inner.change_type)

    inline_body, inline_reason = _covering_inline(text)
    if inline_reason:
        return _review(inline_reason)
    if inline_body is not None:
        if not inline_body.strip():
            return _review("empty")
        inner = _normalize(inline_body.strip())
        if inner.action == "review":
            return inner
        change_type = inner.change_type
        if change_type in ("unchanged", "wrap_plain", "strip_inline"):
            change_type = "strip_inline"
        return _finish(text, inner.text, change_type)

    single = _single_env(text)
    if single:
        return _normalize_env(text, single[0], single[1])

    if "\\begin{" in text or "\\end{" in text:
        return _review("unknown_structure")
    if has_bare_alignment(text):
        return _review("stripped_alignment")
    tokens, error = _scan_delimiters(text)
    if error:
        return _review(error)
    if tokens:
        return _review("residual_delimiter")
    return _finish(text, _wrap(text), "wrap_plain")


def _peel_display(text):
    matched = _ONE_DISPLAY_RE.fullmatch(text)
    if not matched:
        return text, False
    body = matched.group(1) if matched.group(1) is not None else matched.group(2)
    return (body or "").strip(), True


def _normalize_blocks(text, blocks):
    parts = []
    types = []
    for body in blocks:
        if not body:
            return _review("empty")
        inner = _normalize(body)
        if inner.action == "review":
            return inner
        parts.append(inner.text)
        types.append(inner.change_type)
    joined = "\n\n".join(parts)
    real = [item for item in types if item != "unchanged"]
    if not real:
        change_type = "unchanged"
    elif len(set(real)) == 1:
        change_type = real[0]
    else:
        change_type = "mixed"
    return _finish(text, joined, change_type)


def _delimiter_problem(text):
    tokens, error = _scan_delimiters(text)
    if error:
        return error
    if tokens:
        return "residual_delimiter"
    return None


def _normalize_env(text, env, body):
    if env in _KEEP_SHELL or env in _CONVERT:
        problem = _delimiter_problem(body)
        if problem:
            return _review(problem)
    if env in _KEEP_SHELL:
        kept = "\\begin{" + env + "}" + body + "\\end{" + env + "}"
        return _finish(text, _wrap(kept), "keep_environment")
    if env in _CONVERT:
        dest = _CONVERT[env]
        converted = "\\begin{" + dest + "}" + body + "\\end{" + dest + "}"
        change_type = "convert_" + env.rstrip("*")
        return _finish(text, _wrap(converted), change_type)
    if env in _STRIP_SHELL:
        preserved = body.strip()
        if not preserved:
            return _review("empty")
        if has_bare_alignment(preserved):
            return _review("stripped_alignment")
        inner = _normalize(preserved)
        if inner.action == "review":
            return inner
        change_type = inner.change_type
        if change_type not in ("convert_align", "convert_gather",
                               "keep_environment", "mixed"):
            change_type = "strip_equation"
        return _finish(text, inner.text, change_type)
    return _review("unknown_structure")


def equation_display_ok(answer):
    """True when the answer is already display math and needs no rewrite.

    Wrapper whitespace may differ (``$$x$$`` and ``$$\\nx\\n$$``). A rewrite
    that would change the math, including align-to-aligned or a bare ``&``,
    is not accepted.
    """
    text = (answer or "").strip()
    if not text:
        return False
    result = normalize_math(text)
    if result.action == "review":
        return False
    before = _display_blocks(text)
    after = _display_blocks(result.text)
    if not before or not after or len(before) != len(after):
        return False
    for left, right in zip(before, after):
        if left != right:
            return False
        if (not left or has_top_display_env(left) or has_bare_alignment(left)
                or env_balance_error(left) or not _brace_balanced(left)):
            return False
        _tokens, delimiter_error = _scan_delimiters(left)
        if delimiter_error or _tokens:
            return False
    return True
