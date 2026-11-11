#!/usr/bin/env python3
"""Check the hard size limits on source files. Run by pre-commit and CI.

The limits are the constants below.

Python: module docstring, file and function length. Python complexity and arguments are Ruff's C901 and PLR0913.
C, C++ and CUDA: file and function length, complexity and arguments by a scan with comments and strings removed.
It counts branches as C901 does (if, for, while, case, catch), not boolean operators, and the parameters with the
object-like macros of the file's folder expanded (the definition with the most parameters, where there are two).

An exception needs a one-line reason beside it: a comment holding ``size-limit-ok: <reason>`` on the line of the
function or the line above it (in C, the line above the declaration's first line, so above ``template`` or ``extern``),
or in the first lines of the file for a file or module docstring limit.
Takes files as arguments, or every tracked file with none. Skips experiments/. Exits 1 when a limit is broken.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

MAX_DOCSTRING_LINES = 15
MAX_FILE_LINES = 800
# Not 80: that invited mechanical splits, and a step that reads as one ordered operation stays whole.
MAX_FUNCTION_LINES = 150
MAX_COMPLEXITY = 15
MAX_ARGUMENTS = 12
FILE_EXCEPTION_LINES = 5  # a file-wide exception must sit this close to the top, where a reader sees it

PYTHON_SUFFIXES = {".py", ".pyi"}
C_FAMILY_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
SKIPPED_FOLDER = "experiments"  # scripts there need only format, lint and a header

EXCEPTION = re.compile(r"size-limit-ok:\s*\S")
C_NOISE = re.compile(r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'""", re.DOTALL)
C_PREPROCESSOR = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.MULTILINE)
C_BRANCH = re.compile(r"\b(?:if|for|while|case|catch)\b")
C_SIGNATURE_END = re.compile(r"\)\s*(?:(?:const|noexcept|override|final|mutable)\s*)*(?:->[^;{}()]*)?$")
C_CONTROL_WORDS = {"if", "for", "while", "switch", "catch", "sizeof", "return"}
C_WORD_BEFORE = re.compile(r"([A-Za-z_~][\w:~]*)\s*$")
C_OBJECT_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(?![\w(])[ \t]*((?:[^\n]*\\\n)*[^\n]*)", re.MULTILINE)
C_OPERATOR_BEFORE = re.compile(r"([\w:]*operator\s*(?:\(\)|\[\]|[^\s\w()]+))\s*$")
C_FUNCTION_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(", re.MULTILINE)
C_DECLARATION_START = re.compile(r"[;{}]")  # what ends the code before a declaration
UNSUPPORTED = "<unsupported declaration>"  # a definition whose name or parameters the scan cannot read


def _blank(match: re.Match[str]) -> str:
    """Return the matched text with everything but newlines removed, so line numbers stay right."""
    return "\n" * match.group().count("\n")


def _has_exception(lines: list[str], line_number: int) -> bool:
    """Say whether the 1-based line, or the line above it, holds an exception with a reason."""
    return any(EXCEPTION.search(lines[i]) for i in (line_number - 1, line_number - 2) if 0 <= i < len(lines))


def _file_has_exception(lines: list[str]) -> bool:
    """Say whether the top of the file holds an exception with a reason."""
    return any(EXCEPTION.search(line) for line in lines[:FILE_EXCEPTION_LINES])


def _check_file_length(path: Path, lines: list[str]) -> list[str]:
    """Return the failure for a file over the line limit, unless it holds an exception."""
    if len(lines) > MAX_FILE_LINES and not _file_has_exception(lines):
        return [f"{path}:1: file has {len(lines)} lines, limit {MAX_FILE_LINES}"]
    return []


def check_python(path: Path, text: str) -> list[str]:
    """Check one Python file.

    Parameters
    ----------
    path : pathlib.Path
        The file, used in messages.
    text : str
        Its contents.

    Returns
    -------
    list of str
        One message per broken limit.
    """
    lines = text.splitlines()
    failures = _check_file_length(path, lines)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [*failures, f"{path}:{error.lineno}: cannot parse: {error.msg}"]
    docstring = ast.get_docstring(tree, clean=False)
    if docstring is not None:
        count = docstring.count("\n") + 1
        if count > MAX_DOCSTRING_LINES and not _file_has_exception(lines):
            failures.append(f"{path}:1: module docstring has {count} lines, limit {MAX_DOCSTRING_LINES}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > MAX_FUNCTION_LINES and not _has_exception(lines, node.lineno):
                failures.append(f"{path}:{node.lineno}: {node.name} has {length} lines, limit {MAX_FUNCTION_LINES}")
    return failures


def _signature(head: str) -> tuple[str, int, int] | None:
    """Return the name and parameter list of the function whose signature ends this text, or None if it is not one.

    Parameters
    ----------
    head : str
        The code before an opening brace.

    Returns
    -------
    tuple or None
        The word before the parameter list (past an explicit specialisation's ``<...>``), unless that word is a control
        keyword, and the positions of the list's opening and closing parentheses. The word is UNSUPPORTED where the
        text ends like a signature but no name stands before the list (a function returning a function pointer).
    """
    match = C_SIGNATURE_END.search(head)
    if match is None:
        return None
    depth = 0
    for position in range(match.start(), -1, -1):
        if head[position] == ")":
            depth += 1
        elif head[position] == "(":
            depth -= 1
            if depth == 0:
                before = _strip_template_arguments(head[:position])
                word = C_OPERATOR_BEFORE.search(before) or C_WORD_BEFORE.search(before)
                if word is not None and word.group(1) in C_CONTROL_WORDS:
                    return None
                return (UNSUPPORTED if word is None else word.group(1)), position, match.start()
    return None


def _strip_template_arguments(text: str) -> str:
    """Return the text without one balanced ``<...>`` at its end (a specialisation's template arguments)."""
    text = text.rstrip()
    if not text.endswith(">"):
        return text
    depth = 0
    for position in range(len(text) - 1, -1, -1):
        depth += (text[position] == ">") - (text[position] == "<")
        if depth == 0:
            return text[:position]
    return text


def c_functions(code: str) -> list[tuple[str, int, int, int, str]]:
    """Find the outermost function bodies in C-family code with comments, strings and macros removed.

    Parameters
    ----------
    code : str
        The cleaned source.

    Returns
    -------
    list of tuple
        Name, position of the declaration's first line, position of the name's line, position after the closing
        brace and the parameter list's text, per function.
    """
    functions = []
    depth = 0
    current: tuple[str, int, int, int, str] | None = None
    for match in re.finditer(r"[{}]", code):
        if match.group() == "{":
            if current is None:
                signature = _signature(code[: match.start()])
                if signature is not None:
                    name, opening, closing = signature
                    ends = [m.end() for m in C_DECLARATION_START.finditer(code, 0, opening)]
                    head = code[ends[-1] if ends else 0 : opening]
                    first = opening - len(head.lstrip())
                    line = code.rfind("\n", 0, code.rfind(name, 0, opening) if name != UNSUPPORTED else opening) + 1
                    current = (name, code.rfind("\n", 0, first) + 1, line, depth, code[opening + 1 : closing])
            depth += 1
        else:
            depth -= 1
            if current is not None and depth == current[3]:
                functions.append((current[0], current[1], current[2], match.end(), current[4]))
                current = None
    return functions


def c_macros(texts: list[str]) -> dict[str, list[str]]:
    """Return the macros the texts define, by name, continuation lines joined.

    An object-like macro maps to its bodies; a function-like one is listed as ``NAME(`` with none, since a parameter
    list that uses one cannot be counted.
    """
    macros: dict[str, list[str]] = {}
    for text in texts:
        clean = C_NOISE.sub(_blank, text)
        for name, body in C_OBJECT_MACRO.findall(clean):
            macros.setdefault(name, []).append(body.replace("\\\n", " "))
        for name in C_FUNCTION_MACRO.findall(clean):
            macros[f"{name}("] = []
    return macros


def _top_level_commas(text: str) -> int:
    """Return the commas outside any parentheses, brackets, braces or template angle brackets."""
    depth, commas = 0, 0
    for character in text:
        depth = max(0, depth + (character in "([{<") - (character in ")]}>"))
        commas += character == "," and depth == 0
    return commas


def c_parameter_count(parameters: str, macros: dict[str, list[str]]) -> int | None:
    """Return the parameters of a list, its object-like macros expanded (the definition with the most, of several).

    Parameters
    ----------
    parameters : str
        The text between a function's parentheses.
    macros : dict
        Macro bodies by name (``c_macros``).

    Returns
    -------
    int or None
        The number of parameters; 0 for ``()`` and ``(void)``; None where the list calls a function-like macro.
    """
    if any(f"{name}(" in macros for name in re.findall(r"\b(\w+)\s*\(", parameters)):
        return None
    for _ in range(len(macros) + 1):  # each pass expands one level of nesting
        expanded = re.sub(
            r"\b\w+\b",
            lambda m: max(macros[m.group()], key=_top_level_commas) if m.group() in macros else m.group(),
            parameters,
        )
        if expanded == parameters:
            break
        parameters = expanded
    text = parameters.strip()
    if not text or text == "void":
        return 0
    return _top_level_commas(text) + 1


def check_c_family(path: Path, text: str, macros: dict[str, list[str]] | None = None) -> list[str]:
    """Check one C, C++ or CUDA file.

    Parameters
    ----------
    path : pathlib.Path
        The file, used in messages.
    text : str
        Its contents.
    macros : dict, optional
        Object-like macros its parameter lists may use (``c_macros``); the file's own if None.

    Returns
    -------
    list of str
        One message per broken limit.
    """
    lines = text.splitlines()
    failures = _check_file_length(path, lines)
    macros = c_macros([text]) if macros is None else macros
    code = C_PREPROCESSOR.sub(_blank, C_NOISE.sub(_blank, text))
    for name, start, name_start, end, parameters in c_functions(code):
        first_line = code.count("\n", 0, start) + 1
        name_line = code.count("\n", 0, name_start) + 1
        arguments = c_parameter_count(parameters, macros)
        if name == UNSUPPORTED or arguments is None:  # never certified, an exception or not
            failures.append(f"{path}:{first_line}: unsupported declaration: the scan cannot read its name or count its "
                            "parameters; write it in a plain form")  # fmt: skip
            continue
        if any(_has_exception(lines, line) for line in range(first_line, name_line + 1)):
            continue
        body = code[name_start:end]
        length = body.count("\n") + 1
        complexity = 1 + len(C_BRANCH.findall(body))
        if length > MAX_FUNCTION_LINES:
            failures.append(f"{path}:{first_line}: {name} has {length} lines, limit {MAX_FUNCTION_LINES}")
        if complexity > MAX_COMPLEXITY:
            failures.append(f"{path}:{first_line}: {name} has complexity {complexity}, limit {MAX_COMPLEXITY}")
        if arguments > MAX_ARGUMENTS:
            failures.append(f"{path}:{first_line}: {name} has {arguments} arguments, limit {MAX_ARGUMENTS}")
    return failures


def tracked_files() -> list[Path]:
    """Return the existing files git tracks in the current directory."""
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [Path(line) for line in result.stdout.splitlines() if Path(line).is_file()]


def main(arguments: list[str]) -> int:
    """Check the named files, or every tracked file.

    Parameters
    ----------
    arguments : list of str
        File paths.

    Returns
    -------
    int
        0 when every limit holds, else 1.
    """
    paths = [Path(argument) for argument in arguments] or tracked_files()
    failures = []
    for path in paths:
        if SKIPPED_FOLDER in path.parts:
            continue
        if path.suffix in PYTHON_SUFFIXES:
            failures += check_python(path, path.read_text(encoding="utf-8"))
        elif path.suffix in C_FAMILY_SUFFIXES:
            folder = sorted(p for p in path.parent.iterdir() if p.suffix in C_FAMILY_SUFFIXES)
            macros = c_macros([p.read_text(encoding="utf-8") for p in folder])
            failures += check_c_family(path, path.read_text(encoding="utf-8"), macros)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
