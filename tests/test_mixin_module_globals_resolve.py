"""Structural guard for the whole CLASS of bug behind MaintenanceMixin.backfill_max_deviation's
missing `text` import (C2 Task 5 / task-5-review.md): a method moved into a mixin file carries
its OWN code unchanged (an AST-identity proof confirms that), but every bare global name inside
it is resolved against the DEFINING MODULE's globals at call time -- not against manager.py's,
even though these classes are all later combined into one DatabaseManager. A name the mixin
module forgot to import is invisible to an AST comparison (which only looks at the function's own
body) and, if the method has no caller anywhere in the app or the test suite, invisible to the
whole gate too -- exactly how `text` stayed missing from database/maintenance.py through every
gate run in task-5-report.md.

This inspects real bytecode (`dis`), not source text: for every function defined directly on each
mixin class (and on DatabaseManager itself), every LOAD_GLOBAL / LOAD_NAME instruction -- in the
function's own code AND in any nested code object (a closure, a lambda, a comprehension/genexpr,
each of which compiles to its own code object) -- must resolve in that function's own
`__globals__` (the module it was actually defined in) or in builtins. A name that doesn't is
exactly the shape of bug this file exists to catch: it will raise NameError the moment that branch
of the method actually runs, regardless of how clean its AST looks next to the pre-move version.
"""
import builtins
import dis
import importlib
import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_BUILTIN_NAMES = frozenset(vars(builtins))

# (module dotted path, class name) for every mixin this refactor produced, plus
# DatabaseManager itself (whatever it still defines directly, post-move).
_CLASSES = [
    ("laser_trim_analyzer.database.migrations", "MigrationsMixin"),
    ("laser_trim_analyzer.database.specs", "SpecsMixin"),
    ("laser_trim_analyzer.database.ft_matching", "FtMatchingMixin"),
    ("laser_trim_analyzer.database.maintenance", "MaintenanceMixin"),
    ("laser_trim_analyzer.database.smoothness", "SmoothnessMixin"),
    ("laser_trim_analyzer.database.manager", "DatabaseManager"),
]


def _collect_global_loads(code: types.CodeType, seen: set) -> set:
    """Every LOAD_GLOBAL/LOAD_NAME argval reachable from `code`, recursing into
    nested code objects (closures, lambdas, comprehensions/genexprs)."""
    if id(code) in seen:
        return set()
    seen.add(id(code))
    names = {instr.argval for instr in dis.get_instructions(code)
             if instr.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _collect_global_loads(const, seen)
    return names


def _unwrap_function(member):
    """A class-`__dict__` value -> its underlying plain function, or None."""
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, types.FunctionType):
        return member
    return None


def unresolved_globals_for_class(cls) -> list:
    """(func_name, missing_name) for every function DIRECTLY defined on `cls`
    (not inherited) that loads a global/name its own module can't supply."""
    problems = []
    for attr_name, member in vars(cls).items():
        func = _unwrap_function(member)
        if func is None:
            continue
        loaded = _collect_global_loads(func.__code__, set())
        missing = sorted(
            n for n in loaded
            if n not in func.__globals__ and n not in _BUILTIN_NAMES
        )
        problems.extend((attr_name, name) for name in missing)
    return problems


def _all_problems():
    problems = []
    for module_name, class_name in _CLASSES:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        for func_name, missing in unresolved_globals_for_class(cls):
            problems.append((module_name, class_name, func_name, missing))
    return problems


def test_every_mixin_and_databasemanager_method_globals_resolve():
    problems = _all_problems()
    assert not problems, (
        "these methods load a name that their own module's globals cannot supply "
        "(and it is not a builtin) -- calling them will raise NameError, the same "
        "way MaintenanceMixin.backfill_max_deviation did when database/maintenance.py "
        "never imported `text` (task-5-review.md); an AST-identity proof cannot see "
        "this, because the missing name lives in the module, not in the function's "
        "own body:\n" +
        "\n".join(f"  {mod}.{cls}.{fn}: {name!r} is not defined"
                   for mod, cls, fn, name in problems)
    )


def _load_mutated_copy(source_path: Path, drop_name: str, tmp_path: Path):
    """A copy of `source_path` with `drop_name` removed from its `from sqlalchemy
    import ...` line specifically (not a blind whole-file replace, so a mention of
    `drop_name` elsewhere -- a docstring, a comment -- is left alone), loaded as a
    fresh module under a throwaway name."""
    lines = source_path.read_text().splitlines(keepends=True)
    mutated_lines = []
    hit = False
    for line in lines:
        if line.startswith("from sqlalchemy import") and drop_name in line:
            hit = True
            names = [n.strip() for n in line.split("import", 1)[1].split(",")]
            names = [n for n in names if n != drop_name]
            line = "from sqlalchemy import " + ", ".join(names) + "\n"
        mutated_lines.append(line)
    assert hit, f"expected a 'from sqlalchemy import ...' line naming {drop_name!r} in {source_path}"
    mutated = "".join(mutated_lines)

    dest = tmp_path / f"mutated_{source_path.name}"
    dest.write_text(mutated)

    mod_name = f"_mutated_globals_check_{source_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, dest)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(mod_name, None)
    return module


def test_mutation_dropping_a_different_import_is_caught_too(tmp_path):
    """Proves this checker isn't just hard-coded to notice `text`: drop `func`
    from smoothness.py's own sqlalchemy import line (a name several of its
    methods use, e.g. update_smoothness_tracks / get_smoothness_stats) and
    confirm the same checker goes red, naming `func`."""
    import laser_trim_analyzer.database.smoothness as smoothness_mod

    source_path = Path(smoothness_mod.__file__)
    mutated_module = _load_mutated_copy(source_path, "func", tmp_path)

    problems = unresolved_globals_for_class(mutated_module.SmoothnessMixin)
    assert problems, "dropping `func` from smoothness.py's import should be caught"
    assert any(name == "func" for _fn, name in problems), problems
