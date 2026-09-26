"""Every module imports on Python 3.11-3.13, not only on 3.14 (review of ingest-speed Tasks 11-12).

Python 3.14 evaluates annotations lazily (PEP 649); every earlier version evaluates a function's
parameter and return annotations when the function is DEFINED, and a class body's annotations
when the class is created. So a signature naming a class defined further down its module imports
fine on 3.14 and raises NameError on 3.11-3.13 -- which is how `core/model_stats.py` (imported by
the database manager, so by every page) came to refuse to import anywhere but 3.14 while
run_v6.bat said "3.12+". The suite runs on 3.14, where the bug cannot show, so this reads the
source instead: no annotation Python evaluates at definition time may name something its module
binds only later (or only under `if TYPE_CHECKING:`), unless the module defers annotations.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "laser_trim_analyzer"


def _defers_annotations(tree) -> bool:
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body)


def _bindings(tree):
    """(module-level name -> the first line that binds it, names bound only for type checkers)."""
    first, typing_only = {}, set()

    def bound_by(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return [(a.asname or a.name).split(".")[0] for a in node.names]
        return []

    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for inner in node.body:
                typing_only.update(bound_by(inner))
            continue
        for name in bound_by(node):
            first.setdefault(name, node.lineno)
    return first, typing_only - set(first)


def _evaluated_at_definition(node):
    """The annotations Python before 3.14 evaluates when `node` is defined."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if arg is not None and arg.annotation is not None:
                yield arg.annotation
        if node.returns is not None:
            yield node.returns
    elif isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                yield item.annotation
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from _evaluated_at_definition(item)


def forward_references(path: Path):
    """Every unquoted name an import-time annotation of `path` evaluates before it exists."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if _defers_annotations(tree):
        return []
    first, typing_only = _bindings(tree)
    out = []
    for node in tree.body:
        for annotation in _evaluated_at_definition(node):
            for name in ast.walk(annotation):
                if not isinstance(name, ast.Name):
                    continue
                bound = first.get(name.id)
                # a class's own name is not bound while its body runs: >= for a class
                later = bound is not None and (bound > node.lineno or (
                    isinstance(node, ast.ClassDef) and bound == node.lineno))
                if later or name.id in typing_only:
                    where = f"line {bound}" if bound else "only under TYPE_CHECKING"
                    out.append(f"{path.relative_to(SRC.parent)}:{name.lineno}: {name.id} "
                               f"(bound {where})")
    return out


def test_no_module_evaluates_an_annotation_before_its_name_exists():
    problems = [p for path in sorted(SRC.rglob("*.py")) for p in forward_references(path)]
    assert not problems, ("these annotations raise NameError on import under Python 3.11-3.13 "
                          "-- quote them, or defer the module's annotations:\n  "
                          + "\n  ".join(problems))


def test_the_check_sees_the_shape_that_broke_model_stats(tmp_path):
    """The check is not vacuous: model_stats' own shape before the fix is caught, a quoted one is
    not, and neither is a module that defers its annotations."""
    bad = tmp_path / "laser_trim_analyzer" / "bad.py"
    bad.parent.mkdir()
    source = ("from typing import Optional\n\n"
              "def lot_line(verdict: Optional[LotVerdict]) -> str:\n    return ''\n\n"
              "class LotVerdict:\n    pass\n")
    bad.write_text(source)
    import test_imports_before_314 as me
    real = me.SRC
    me.SRC = tmp_path / "laser_trim_analyzer"
    try:
        assert me.forward_references(bad) == ["laser_trim_analyzer/bad.py:3: LotVerdict (bound line 6)"]
        bad.write_text(source.replace("Optional[LotVerdict]", 'Optional["LotVerdict"]'))
        assert me.forward_references(bad) == []
        bad.write_text("from __future__ import annotations\n" + source)
        assert me.forward_references(bad) == []
    finally:
        me.SRC = real
