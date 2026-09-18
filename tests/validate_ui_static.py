"""Lightweight static checks for the Unity IMGUI and localization surfaces.

Unity is not available in the Python test environment, so this intentionally
checks only source-level contracts that can regress without opening the Editor:

* Begin*/End* layout groups must close in LIFO order inside each method.
* Literal strings passed through ``TrLiteral``/``TrFormat`` should have a
  Japanese dictionary entry.
* Direct user-facing IMGUI literals are reported for manual localization
  review.  Icons and interpolated/dynamic strings are not treated as failures.

The default invocation is a report-only check (exit code 0).  ``--strict`` is
available for CI and returns 1 when malformed layout groups or missing
localization entries are found.  This keeps the existing Python test command
useful while still making protected Unity UI blockers explicit.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_FILES = (
    ROOT / "Editor" / "TexMotionWindow.cs",
    ROOT / "Editor" / "Motion" / "MotionTimelineEditorWindow.cs",
)
LOCALIZATION_FILE = ROOT / "Editor" / "Localization" / "TexMotionLocalization.cs"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT).as_posix()}:{self.line}: {self.message}"


# The small helper methods in TexMotionWindow intentionally split a Begin call
# from its EndVertical/EndHorizontal at the call site.  Treat those helpers as
# their underlying layout group for the source-level check.
CUSTOM_GROUPS = {
    "BeginCard": "Vertical",
    "BeginNestedCard": "Vertical",
    "BeginInlineCard": "Horizontal",
    "BeginNestedRow": "Horizontal",
}

GROUP_RE = re.compile(
    r"(?:(?P<owner>EditorGUILayout|GUILayout|GUI|Handles|_previewUtility)\."
    r"(?P<kind>Begin|End)(?P<group>[A-Za-z]+)|"
    r"(?P<custom>BeginCard|BeginNestedCard|BeginInlineCard|BeginNestedRow)\s*\()"
)

# Access modifiers make this conservative enough to identify method starts
# without trying to parse all of C# (which is deliberately outside this
# lightweight check's scope).
METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|async|virtual|override|sealed|new)\s+)+"
    r"[A-Za-z_][\w<>\[\],.? ]*\s+[A-Za-z_]\w*\s*\([^;{}]*\)"
)

VISIBLE_CALL_RE = re.compile(
    r"(?P<call>"
    r"GUILayout\.(?:Button|Toggle)|"
    r"EditorGUILayout\.(?:LabelField|HelpBox|TextField|ObjectField|EnumPopup|Slider|IntSlider|FloatField|Toggle|ToggleLeft|Popup|IntPopup)|"
    r"EditorGUI\.LabelField|GUI\.(?:Label|Button|Toggle)|"
    r"new\s+GUIContent|EditorUtility\.DisplayDialog"
    r")\s*\(\s*(?P<literal>\"(?:\\.|[^\"\\])*\")"
)

TR_LITERAL_RE = re.compile(
    r"TexMotionLocalization\.Tr(?:Literal|Format)\s*\(\s*(?P<literal>\"(?:\\.|[^\"\\])*\")"
)


def decode_literal(token: str) -> str:
    """Decode a C#-style quoted literal sufficiently for dictionary matching."""

    try:
        # C# and Python share the escapes relevant to the localization keys in
        # this project.  Keep the original token if a future key uses an
        # unsupported escape rather than making the validator fail collection.
        return ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return token[1:-1]


def _mask_comments_and_strings(source: str) -> str:
    """Mask comments/strings while preserving line count for method scanning.

    Interpolated expressions are not evaluated; the scanner only needs to keep
    braces in ordinary string content from disturbing method boundaries.  The
    source-level layout pass is intentionally conservative and independently
    reports the known nested-group ordering even when an interpolation contains
    a nested quoted expression.
    """

    chars = list(source)
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] == "/" and i + 1 < n and chars[i + 1] == "/":
            i += 2
            while i < n and chars[i] != "\n":
                chars[i] = " "
                i += 1
            continue
        if chars[i] == "/" and i + 1 < n and chars[i + 1] == "*":
            chars[i] = chars[i + 1] = " "
            i += 2
            while i + 1 < n and not (chars[i] == "*" and chars[i + 1] == "/"):
                if chars[i] != "\n":
                    chars[i] = " "
                i += 1
            if i + 1 < n:
                chars[i] = chars[i + 1] = " "
                i += 2
            continue
        if chars[i] in ('"', "'"):
            quote = chars[i]
            # Preserve quote delimiters, but mask the content.  This is enough
            # for brace matching in normal C# strings.
            i += 1
            while i < n:
                if chars[i] == "\\":
                    if chars[i] != "\n":
                        chars[i] = " "
                    if i + 1 < n:
                        if chars[i + 1] != "\n":
                            chars[i + 1] = " "
                        i += 2
                    else:
                        i += 1
                    continue
                if chars[i] == quote:
                    i += 1
                    break
                if chars[i] != "\n":
                    chars[i] = " "
                i += 1
            continue
        i += 1
    return "".join(chars)


def _method_ranges(source: str) -> list[tuple[str, int, int]]:
    """Return (method name, first line, exclusive last line) ranges."""

    masked = _mask_comments_and_strings(source)
    lines = masked.splitlines()
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = METHOD_RE.match(line)
        if not match:
            continue
        name_match = re.search(r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$", match.group(0))
        if name_match:
            starts.append((name_match.group(1), index))
    ranges = []
    for pos, (name, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        ranges.append((name, start, end))
    return ranges


def find_layout_findings(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    findings: list[Finding] = []
    for method, start, end in _method_ranges(source):
        # These are the helper implementations, not call sites.  They leave
        # their layout group open by design for the caller to close.
        if method in CUSTOM_GROUPS:
            continue
        stack: list[tuple[str, int]] = []
        for index in range(start, end):
            for match in GROUP_RE.finditer(lines[index]):
                custom = match.group("custom")
                if custom:
                    group = CUSTOM_GROUPS[custom]
                    is_end = False
                else:
                    group = match.group("group")
                    is_end = match.group("kind") == "End"
                if not is_end:
                    stack.append((group, index + 1))
                    continue
                if stack and stack[-1][0] == group:
                    stack.pop()
                    continue
                # If the matching group is still open below the top, this is a
                # deterministic LIFO violation (rather than a branch-sensitive
                # missing group).  Remove it to avoid cascading duplicate lines.
                matching_index = next(
                    (i for i in range(len(stack) - 1, -1, -1) if stack[i][0] == group),
                    None,
                )
                if matching_index is not None:
                    findings.append(
                        Finding(
                            path,
                            index + 1,
                            f"{method}: End{group} closes before End{stack[-1][0]} (Begin at line {stack[matching_index][1]})",
                        )
                    )
                    del stack[matching_index]
                else:
                    # An End call in a return-only branch may be balanced by a
                    # second End call on the fall-through path.  Without a
                    # control-flow parser, an unmatched close is therefore
                    # advisory rather than a deterministic layout defect.
                    continue
        for group, line in stack:
            findings.append(Finding(path, line, f"{method}: Begin{group} has no matching End{group}"))
    return findings


CONST_RE = re.compile(
    r"public\s+const\s+string\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<literal>\"(?:\\.|[^\"\\])*\")"
)
DICT_ENTRY_RE = re.compile(
    r"\{\s*(?P<key>[A-Za-z_]\w*|\"(?:\\.|[^\"\\])*\")\s*,\s*"
    r"(?P<value>\"(?:\\.|[^\"\\])*\")"
)


def _dictionary_body(source: str, name: str) -> str:
    """Return the initializer body for a named string dictionary."""

    marker = f"Dictionary<string, string> {name}"
    marker_index = source.find(marker)
    if marker_index < 0:
        return ""
    start = source.find("{", marker_index)
    if start < 0:
        return ""

    # A simple ``source.find("};")`` stops early when a format string contains
    # the same sequence (for example ``"{0}; fallback"``).  Walk the
    # initializer with C# string/comment awareness and use brace depth to find
    # the dictionary's closing brace instead.
    depth = 0
    i = start
    n = len(source)
    in_string = False
    quote = ""
    while i < n:
        ch = source[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            newline = source.find("\n", i + 2)
            i = n if newline < 0 else newline + 1
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            close = source.find("*/", i + 2)
            i = n if close < 0 else close + 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:i]
        i += 1
    return ""


def localization_catalog() -> tuple[set[str], dict[str, str]]:
    """Resolve canonical constants and legacy English literals to Japanese keys."""

    source = LOCALIZATION_FILE.read_text(encoding="utf-8-sig")
    constants = {
        match.group("name"): decode_literal(match.group("literal"))
        for match in CONST_RE.finditer(source)
    }

    def resolve_key(token: str) -> str:
        token = token.strip()
        if token.startswith('"'):
            return decode_literal(token)
        return constants.get(token, token)

    japanese_entries: set[str] = set()
    for match in DICT_ENTRY_RE.finditer(_dictionary_body(source, "JapaneseText")):
        japanese_entries.add(resolve_key(match.group("key")))

    english_to_canonical: dict[str, str] = {}
    for match in DICT_ENTRY_RE.finditer(_dictionary_body(source, "EnglishText")):
        english_to_canonical[decode_literal(match.group("value"))] = resolve_key(match.group("key"))

    # TrLiteral still accepts legacy English display strings. Include those
    # labels alongside canonical keys when checking call sites.
    covered_literals = set(japanese_entries)
    covered_literals.update(
        english for english, canonical in english_to_canonical.items() if canonical in japanese_entries
    )
    return covered_literals, english_to_canonical


def japanese_keys() -> set[str]:
    if not LOCALIZATION_FILE.exists():
        return set()
    return localization_catalog()[0]


def find_missing_localization(path: Path, keys: set[str]) -> list[Finding]:
    source = path.read_text(encoding="utf-8-sig")
    findings: list[Finding] = []
    for match in TR_LITERAL_RE.finditer(source):
        value = decode_literal(match.group("literal"))
        if value not in keys:
            line = source.count("\n", 0, match.start()) + 1
            findings.append(Finding(path, line, f"missing Japanese localization key: {value!r}"))
    return findings


def direct_ui_literals(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8-sig")
    findings: list[Finding] = []
    for match in VISIBLE_CALL_RE.finditer(source):
        value = decode_literal(match.group("literal"))
        # Dynamic labels and icon-only prefixes are not useful localization
        # findings.  Keep phrases containing at least two ASCII letters.
        if len(re.findall(r"[A-Za-z]", value)) < 2:
            continue
        line = source.count("\n", 0, match.start()) + 1
        findings.append(Finding(path, line, f"direct UI literal: {value!r}"))
    return findings


def validate(strict: bool = False) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    layout: list[Finding] = []
    missing: list[Finding] = []
    direct: list[Finding] = []
    keys = japanese_keys()
    for path in UI_FILES:
        layout.extend(find_layout_findings(path))
        missing.extend(find_missing_localization(path, keys))
        direct.extend(direct_ui_literals(path))

    if not LOCALIZATION_FILE.exists():
        missing.insert(
            0,
            Finding(
                LOCALIZATION_FILE,
                1,
                "localization source is missing; Japanese coverage cannot be evaluated",
            ),
        )

    print("UI static validation")
    print(f"  layout findings: {len(layout)}")
    for finding in layout:
        print(f"    - {finding.format()}")
    print(f"  missing TrLiteral/TrFormat Japanese keys: {len(missing)}")
    for finding in missing[:12]:
        print(f"    - {finding.format()}")
    if len(missing) > 12:
        print(f"    - ... {len(missing) - 12} more")
    print(f"  direct UI literals for manual review: {len(direct)}")
    by_file: dict[Path, int] = {}
    for finding in direct:
        by_file[finding.path] = by_file.get(finding.path, 0) + 1
    for path, count in by_file.items():
        print(f"    - {path.relative_to(ROOT).as_posix()}: {count}")
    for finding in direct[:12]:
        print(f"    - {finding.format()}")
    if len(direct) > 12:
        print(f"    - ... {len(direct) - 12} more")

    if strict and (layout or missing):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on malformed layout or missing localization keys")
    args = parser.parse_args()
    return validate(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
