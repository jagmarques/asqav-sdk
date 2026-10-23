"""Fixture tests for the SDK docstring and comment concision checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_comment_concision.py"
_spec = importlib.util.spec_from_file_location("check_comment_concision", SCRIPT)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checker
_spec.loader.exec_module(checker)

MAX_BLOCK_ROWS = checker.MAX_BLOCK_ROWS
MAX_COMMENT_COLUMNS = checker.MAX_COMMENT_COLUMNS
MAX_DOCSTRING_LINES = checker.MAX_DOCSTRING_LINES
main = checker.main

PKG = Path("python/src/asqav")


def plant(root: Path, name: str, text: str) -> None:
    target = root / PKG / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def docstring(lines: int, args: bool = False) -> str:
    """A docstring that spans exactly `lines` physical lines."""
    body = ["    Summary."]
    if args:
        body += ["", "    Args:", "        key: the key."]
    while len(body) < lines - 1:
        body.append("    more")
    body.append('    """')
    return '    """' + "\n".join(body)[4:] + "\n"


def func(name: str, lines: int, args: bool = False) -> str:
    return f"def {name}(key):\n{docstring(lines, args)}    return key\n"


def comment_rows(count: int) -> str:
    return "".join("# row\n" for _ in range(count)) + "x = 1\n"


def run(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main([str(root)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_six_line_docstring_passes(tmp_path, capsys):
    plant(tmp_path, "client.py", func("sign", MAX_DOCSTRING_LINES))
    assert run(tmp_path, capsys)[0] == 0


# NEGCTL: a planted 7-line docstring in client.py through the real main().
def test_seven_line_docstring_in_client_fails(tmp_path, capsys):
    plant(tmp_path, "client.py", func("sign", MAX_DOCSTRING_LINES + 1))
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert "client.py:" in out
    assert "docstring" in out


def test_eighty_one_char_comment_fails(tmp_path, capsys):
    pad = "a" * (MAX_COMMENT_COLUMNS + 1 - len("# "))
    plant(tmp_path, "x.py", f"y = 1\n# {pad}\n")
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert "x.py:2:comment-line" in out


def test_eighty_char_comment_passes(tmp_path, capsys):
    pad = "a" * (MAX_COMMENT_COLUMNS - len("# "))
    plant(tmp_path, "x.py", f"y = 1\n# {pad}\n")
    assert run(tmp_path, capsys)[0] == 0


def test_trailing_comment_counts_its_column(tmp_path, capsys):
    pad = "a" * (MAX_COMMENT_COLUMNS - len("y = 1  # "))
    plant(tmp_path, "x.py", f"y = 1  # {pad}a\n")
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert "x.py:1:comment-line" in out


def test_three_line_comment_block_fails(tmp_path, capsys):
    plant(tmp_path, "x.py", comment_rows(MAX_BLOCK_ROWS + 1))
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert "x.py:1:comment-block" in out


def test_two_line_block_passes(tmp_path, capsys):
    plant(tmp_path, "x.py", comment_rows(MAX_BLOCK_ROWS))
    assert run(tmp_path, capsys)[0] == 0


def test_trailing_comments_do_not_form_a_block(tmp_path, capsys):
    rows = "".join("x = 1  # row\n" for _ in range(MAX_BLOCK_ROWS + 1))
    plant(tmp_path, "x.py", rows)
    assert run(tmp_path, capsys)[0] == 0


def test_public_args_docstring_passes(tmp_path, capsys):
    plant(tmp_path, "client.py", func("sign", MAX_DOCSTRING_LINES + 3, args=True))
    assert run(tmp_path, capsys)[0] == 0


def test_private_args_docstring_fails(tmp_path, capsys):
    plant(tmp_path, "client.py", func("_sign", MAX_DOCSTRING_LINES + 3, args=True))
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert "client.py:2:docstring" in out


def test_private_module_args_docstring_fails(tmp_path, capsys):
    plant(tmp_path, "_helpers.py", func("sign", MAX_DOCSTRING_LINES + 3, args=True))
    assert run(tmp_path, capsys)[0] == 1


def test_method_of_private_class_args_docstring_fails(tmp_path, capsys):
    method = func("sign", MAX_DOCSTRING_LINES + 3, args=True)
    indented = "".join(f"    {line}\n" for line in method.splitlines())
    plant(tmp_path, "client.py", f"class _Signer:\n{indented}")
    assert run(tmp_path, capsys)[0] == 1


def test_method_of_public_class_args_docstring_passes(tmp_path, capsys):
    method = func("sign", MAX_DOCSTRING_LINES + 3, args=True)
    indented = "".join(f"    {line}\n" for line in method.splitlines())
    plant(tmp_path, "client.py", f"class Signer:\n{indented}")
    assert run(tmp_path, capsys)[0] == 0


def test_long_module_docstring_with_args_fails(tmp_path, capsys):
    body = "\n".join(["Summary.", "", "Args:", "    key: the key."] + ["more"] * 5)
    plant(tmp_path, "client.py", f'"""{body}\n"""\n')
    assert run(tmp_path, capsys)[0] == 1


def test_verifier_dir_excluded(tmp_path, capsys):
    plant(tmp_path, "verifier/check.py", func("sign", MAX_DOCSTRING_LINES + 2))
    assert run(tmp_path, capsys)[0] == 0


def test_verifier_prefix_not_excluded(tmp_path, capsys):
    plant(tmp_path, "verifier_extra.py", func("sign", MAX_DOCSTRING_LINES + 2))
    assert run(tmp_path, capsys)[0] == 1


def test_unparsable_file_exits_2(tmp_path, capsys):
    plant(tmp_path, "broken.py", "def (:\n")
    code, _, err = run(tmp_path, capsys)
    assert code == 2
    assert "broken.py" in err


def test_unreadable_file_exits_2(tmp_path, capsys):
    target = tmp_path / PKG / "binary.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xff\xfe\x00bad")
    code, _, err = run(tmp_path, capsys)
    assert code == 2
    assert "binary.py" in err


def test_missing_source_dir_exits_2(tmp_path, capsys):
    code, _, err = run(tmp_path, capsys)
    assert code == 2
    assert str(PKG.as_posix()) in err


def test_hash_in_string_ignored(tmp_path, capsys):
    pad = "a" * (MAX_COMMENT_COLUMNS + 5)
    rows = "\n".join(f'x{i} = "# {pad}"' for i in range(MAX_BLOCK_ROWS + 1))
    plant(tmp_path, "x.py", rows + "\n")
    assert run(tmp_path, capsys)[0] == 0


# NEGCTL per condition: one planted row each, failing only on its own kind.
@pytest.mark.parametrize(
    ("name", "text", "kind"),
    [
        ("long.py", "# " + "a" * MAX_COMMENT_COLUMNS + "\n", "comment-line"),
        ("block.py", comment_rows(MAX_BLOCK_ROWS + 1), "comment-block"),
        ("client.py", func("_sign", MAX_DOCSTRING_LINES + 3, args=True), "docstring"),
    ],
)
def test_each_condition_fails_only_on_its_own_kind(tmp_path, capsys, name, text, kind):
    plant(tmp_path, name, text)
    code, out, _ = run(tmp_path, capsys)
    assert code == 1
    assert {line.rsplit(":", 1)[1] for line in out.splitlines()} == {kind}


REPO = SCRIPT.parents[2]


# NEGCTL: test_seven_line_docstring_in_client_fails plants client.py findings.
def test_client_py_is_concise():
    found = [f for f in checker.scan(REPO) if f.path == "python/src/asqav/client.py"]
    assert found == []


# NEGCTL: test_seven_line_docstring_in_client_fails plants client.py findings.
def test_core_modules_are_concise():
    found = [f for f in checker.scan(REPO) if "/extras/" not in f.path]
    assert found == []
