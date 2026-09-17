#!/usr/bin/env python3
"""ATP-versus-station linearity spec audit.

The customer's Acceptance Test Procedure (ATP) data sheets carry the linearity
tolerance band a model is *supposed* to be graded against.  The trim station and
the final-test station each write their own limit columns into the files the
analyzer ingests.  Nothing in the pipeline has ever checked that the three agree.

This script does exactly that, per model and per station:

  1. Index the ATP document library and pull every embedded linearity table.
  2. Normalise each table to {angle: tolerance} plus a band descriptor.
  3. Read what the stations actually graded to, from the limit arrays stored on
     ``track_results`` (trim) and ``final_test_tracks`` (final test).
  4. Compare, and emit a verdict per (model, station).
  5. Cross-check the ATP prose specs (resistance / electrical angle / linearity
     type) against the ``model_specs`` table.
  6. Rank everything by production volume so the models that matter lead.

Everything is read-only.  The database is opened with ``mode=ro`` and the ATP
library is only ever read.  Output goes to ``qa_output/atp_audit/`` (gitignored).

    .venv/bin/python scripts/atp_spec_audit.py data/analysis.db
    .venv/bin/python scripts/atp_spec_audit.py data/analysis.db --self-test

The BIFF8 reader below started life as ``atp_read.py``, a throwaway probe written
during the 2026-09-17 investigation to prove the per-point tables really are
embedded Excel workbooks inside the ``.doc`` files (``ObjectPool/_<id>/Workbook``).
It is reproduced here, extended for ``.docx`` (zip, ``word/embeddings/``) and
``.xlsx`` (openpyxl), so the audit is self-contained.

``olefile`` is deliberately NOT in requirements-pinned.txt -- this is an offline
analysis tool, not part of the app.  Install it into a scratch dir and point
PYTHONPATH at it; the script prints a hint if it is missing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import re
import sqlite3
import struct
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("atp_spec_audit")

# --------------------------------------------------------------------------- #
# Tunables -- every threshold the comparison leans on, in one place.
# --------------------------------------------------------------------------- #

SHAPE_TOL = 0.05        # shape ratios within 5% are "the same shape"
RATIO_TOL = 0.02        # value ratios within 2% across angles are "constant"
INTERP_TOL = 1e-6       # absolute tolerance on the linear-interpolation check
ANGLE_TOL = 0.05        # two angles this close (in station units) are the same point
AXIS_SPAN_TOL = 0.10    # ATP axis and station axis spans within 10% -> direct alignment
                        # (ATP tables routinely omit an end row, so 5% was too tight:
                        #  8275's table prints 5..90 for a station that sweeps 90)
AXIS_GIVE_UP = 0.50     # spans differing by more than 1.5x are different quantities,
                        # not different units -- do not compare angles at all
CLEAN_SCALES = (1.0, 2.0, 5.0, 10.0, 0.5, 0.2, 0.1)

# Folders / filenames we never audit.
SKIP_DIR_RE = re.compile(r"(?:^|/)(archive|archived|obsolete|old)(?:/|$)", re.I)
SKIP_NAME_RE = re.compile(r"(^~\$)|(obsolete)|(\.tmp$)|(\.lnk$)|(\.db$)|(\.jpg$)|(\.jpeg$)|(\.png$)", re.I)

DOC_EXT = {".doc"}
DOCX_EXT = {".docx", ".docm"}
XLSX_EXT = {".xlsx", ".xlsm", ".xltm", ".xlt", ".xls"}
PDF_EXT = {".pdf"}


# =========================================================================== #
# 1. Readers
# =========================================================================== #

def _require_olefile():
    """Import olefile, or explain in one line how to get it."""
    try:
        import olefile  # noqa: F401
        return olefile
    except ImportError:
        print(
            "olefile is required to read .doc ATP sheets and is intentionally NOT "
            "in requirements-pinned.txt -- install it somewhere scratch and point "
            "PYTHONPATH at it, e.g.:  "
            "pip install --target /tmp/pylibs olefile && "
            "PYTHONPATH=/tmp/pylibs:$PWD/src .venv/bin/python scripts/atp_spec_audit.py ...",
            file=sys.stderr,
        )
        return None


# --- BIFF8 record parsing (from the scratch atp_read.py probe) -------------- #

def _unrk(rk: int) -> float:
    rk &= 0xFFFFFFFF
    if rk & 0x02:
        v = rk >> 2
        if v >= 0x20000000:
            v -= 0x40000000
        val = float(v)
    else:
        val = struct.unpack("<d", struct.pack("<Q", (rk & 0xFFFFFFFC) << 32))[0]
    return val / 100.0 if rk & 0x01 else val


def _records(buf: bytes) -> List[Tuple[int, bytes]]:
    """(rectype, payload) pairs, merging CONTINUE (0x3C) into the previous record."""
    i, out = 0, []
    while i + 4 <= len(buf):
        rt, ln = struct.unpack("<HH", buf[i:i + 4])
        i += 4
        data = buf[i:i + ln]
        i += ln
        if rt == 0x003C and out:
            out[-1] = (out[-1][0], out[-1][1] + data)
        else:
            out.append((rt, data))
    return out


def _sst(payload: bytes) -> List[str]:
    """Shared string table -> list of str.  Best effort; stops cleanly on trouble."""
    strings: List[str] = []
    try:
        n = struct.unpack("<i", payload[4:8])[0]
        p = 8
        for _ in range(n):
            if p + 3 > len(payload):
                break
            cch = struct.unpack("<H", payload[p:p + 2])[0]
            p += 2
            flags = payload[p]
            p += 1
            rich = struct.unpack("<H", payload[p:p + 2])[0] if flags & 0x08 else 0
            if flags & 0x08:
                p += 2
            ext = struct.unpack("<i", payload[p:p + 4])[0] if flags & 0x04 else 0
            if flags & 0x04:
                p += 4
            if flags & 0x01:
                s = payload[p:p + cch * 2].decode("utf-16-le", "replace")
                p += cch * 2
            else:
                s = payload[p:p + cch].decode("latin-1", "replace")
                p += cch
            p += rich * 4 + ext
            strings.append(s)
    except Exception:
        pass
    return strings


def biff_cells(workbook_bytes: bytes) -> Dict[Tuple[int, int], Any]:
    """{(row, col): value} for one embedded BIFF8 workbook stream."""
    recs = _records(workbook_bytes)
    sst: List[str] = []
    for rt, d in recs:
        if rt == 0x00FC:
            sst = _sst(d)
            break
    out: Dict[Tuple[int, int], Any] = {}
    for rt, d in recs:
        try:
            if rt == 0x0203 and len(d) >= 14:                 # NUMBER
                r, c = struct.unpack("<HH", d[0:4])
                out[(r, c)] = struct.unpack("<d", d[6:14])[0]
            elif rt == 0x027E and len(d) >= 10:               # RK
                r, c = struct.unpack("<HH", d[0:4])
                out[(r, c)] = _unrk(struct.unpack("<I", d[6:10])[0])
            elif rt == 0x00BD and len(d) >= 6:                # MULRK
                r, c0 = struct.unpack("<HH", d[0:4])
                p, c = 4, c0
                while p + 6 <= len(d) - 2:
                    out[(r, c)] = _unrk(struct.unpack("<I", d[p + 2:p + 6])[0])
                    p += 6
                    c += 1
            elif rt == 0x00FD and len(d) >= 10:               # LABELSST
                r, c = struct.unpack("<HH", d[0:4])
                idx = struct.unpack("<i", d[6:10])[0]
                if 0 <= idx < len(sst):
                    out[(r, c)] = sst[idx]
            elif rt == 0x0006 and len(d) >= 20:               # FORMULA (cached value)
                r, c = struct.unpack("<HH", d[0:4])
                raw = d[6:14]
                if raw[6:8] != b"\xff\xff":
                    out[(r, c)] = struct.unpack("<d", raw)[0]
        except Exception:
            continue
    return out


def doc_embedded_tables(path: Path) -> List[Tuple[str, Dict[Tuple[int, int], Any]]]:
    """Every embedded Excel workbook inside a Word .doc (OLE ObjectPool)."""
    olefile = _require_olefile()
    if olefile is None:
        return []
    ole = olefile.OleFileIO(str(path))
    try:
        found = []
        for entry in ole.listdir(streams=True):
            if len(entry) >= 2 and entry[0] == "ObjectPool" and entry[-1] == "Workbook":
                found.append(("/".join(entry), biff_cells(ole.openstream(entry).read())))
        return found
    finally:
        ole.close()


def docx_embedded_tables(path: Path) -> List[Tuple[str, Dict[Tuple[int, int], Any]]]:
    """Embedded spreadsheets inside a .docx (zip -> word/embeddings/*)."""
    out: List[Tuple[str, Dict[Tuple[int, int], Any]]] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().startswith("word/embeddings/")]
        for n in names:
            data = zf.read(n)
            low = n.lower()
            if low.endswith((".xlsx", ".xlsm")) or data[:4] == b"PK\x03\x04":
                try:
                    out.append((n, _xlsx_bytes_cells(data)))
                except Exception as exc:          # pragma: no cover - corrupt embed
                    log.debug("docx embed %s: %s", n, exc)
            elif data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                olefile = _require_olefile()
                if olefile is None:
                    continue
                import io
                try:
                    ole = olefile.OleFileIO(io.BytesIO(data))
                    try:
                        for entry in ole.listdir(streams=True):
                            if entry[-1] == "Workbook":
                                out.append((f"{n}/{'/'.join(entry)}",
                                            biff_cells(ole.openstream(entry).read())))
                    finally:
                        ole.close()
                except Exception as exc:          # pragma: no cover
                    log.debug("docx OLE embed %s: %s", n, exc)
    return out


def _xlsx_bytes_cells(data: bytes) -> Dict[Tuple[int, int], Any]:
    import io
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        cells: Dict[Tuple[int, int], Any] = {}
        for r, row in enumerate(ws.iter_rows(values_only=True)):
            for c, v in enumerate(row):
                if v is not None and v != "":
                    cells[(r, c)] = v
        return cells
    finally:
        wb.close()


def xlsx_tables(path: Path) -> List[Tuple[str, Dict[Tuple[int, int], Any]]]:
    """Every worksheet of a standalone workbook, as a cell map."""
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    try:
        out = []
        for name in wb.sheetnames:
            ws = wb[name]
            cells: Dict[Tuple[int, int], Any] = {}
            for r, row in enumerate(ws.iter_rows(values_only=True)):
                for c, v in enumerate(row):
                    if v is not None and v != "":
                        cells[(r, c)] = v
            if cells:
                out.append((name, cells))
        return out
    finally:
        wb.close()


def textutil_text(path: Path) -> str:
    """Prose, via the macOS built-in textutil (handles .doc and .docx)."""
    try:
        res = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return res.stdout or ""
    except Exception as exc:                       # pragma: no cover
        log.debug("textutil %s: %s", path, exc)
        return ""


_WP_RE = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)
_WT_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
_ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'"}


def docx_text_from_zip(path: Path) -> str:
    """Body + header + footer text of a .docx, reconstructed from the XML.

    textutil returns nothing at all for some of these sheets, and the MODEL line
    frequently lives in a header rather than the body.  Word also splits a single
    word across many <w:r> runs ("MK 8 87 7" for MK8877), so runs are joined
    *within* a paragraph with no separator and paragraphs become lines --
    replacing every tag with a space would keep the word broken.
    """
    parts: List[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist()
                     if re.fullmatch(r"word/(document|header\d*|footer\d*)\.xml", n)]
            names.sort(key=lambda n: (0 if "document" in n else 1, n))
            for n in names:
                xml = zf.read(n).decode("utf-8", "replace")
                for para in _WP_RE.findall(xml):
                    runs = "".join(_WT_RE.findall(para))
                    for k, v in _ENT.items():
                        runs = runs.replace(k, v)
                    runs = runs.strip()
                    if runs:
                        parts.append(runs)
    except Exception as exc:
        log.debug("docx zip text %s: %s", path, exc)
    return "\n".join(parts)


def xlsx_text(path: Path) -> str:
    """Every string cell of a workbook, one per line.

    Several models' only sheet in the library is a workbook ('6607-Final Test
    Data Rev K.xlsx'), and its MODEL line is an ordinary cell.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    except Exception as exc:
        log.debug("xlsx text %s: %s", path, exc)
        return ""
    lines: List[str] = []
    try:
        for name in wb.sheetnames:
            for row in wb[name].iter_rows(values_only=True):
                for v in row:
                    if isinstance(v, str) and v.strip():
                        lines.append(v.strip())
    except Exception as exc:
        log.debug("xlsx text %s: %s", path, exc)
    finally:
        wb.close()
    return "\n".join(lines)


def pdf_text(path: Path) -> str:
    """Text from a PDF *only if trivially available*.  Otherwise "" (un-audited)."""
    for cmd in (["pdftotext", "-q", str(path), "-"],):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout
        except FileNotFoundError:
            break
        except Exception:
            break
    return ""


# =========================================================================== #
# 2. ATP table normalisation
# =========================================================================== #

_DASHES = {"-", "--", "---", "----", "-----", "n/a", "na", "", "*", "**"}


def parse_tolerance(v: Any) -> Optional[float]:
    """A tolerance cell -> float, or None when the sheet says 'not graded'.

    Handles floats, and text forms such as '+ .0200', '+/- .005', '±0.01', '----'.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return abs(float(v))
    s = str(v).strip()
    if s.lower().replace(" ", "") in _DASHES:
        return None
    s = s.replace("±", "").replace("+/-", "").replace("+/ -", "")
    s = s.replace("+", "").replace("%", "").replace(",", "").strip()
    if not s or s.lower() in _DASHES:
        return None
    try:
        return abs(float(s))
    except ValueError:
        m = re.search(r"-?\d*\.?\d+", s)
        if m:
            try:
                return abs(float(m.group(0)))
            except ValueError:
                return None
        return None


def _num(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace('"', "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _hdr(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower() if isinstance(v, str) else ""


_TOL_RE = re.compile(r"\btol")
_ANGLE_RE = re.compile(r"angle|travel|position|rotation|degree|stroke")
_VALUE_RE = re.compile(r"v\.?\s*r\.?|volt|ratio|output|theo")
_IGNORE_RE = re.compile(r"error|actual|allow|shop|s/n|stamp")


@dataclass
class AtpTable:
    """One normalised per-point linearity table out of one ATP sheet."""
    source: str                       # file path
    object_name: str                  # stream / sheet name inside the file
    angle_header: str = ""
    value_header: str = ""
    tol_header: str = ""
    unit_hint: str = ""               # 'Degrees' / 'Inches' from the "Linearity Table For:" row
    angles: List[float] = field(default_factory=list)      # x, as printed
    tolerances: List[Optional[float]] = field(default_factory=list)
    values: List[Optional[float]] = field(default_factory=list)
    role: str = "unknown"             # 'trim' | 'final_test' | 'unknown'
    role_evidence: str = ""
    role_confidence: str = "low"      # 'high' | 'low'

    # derived
    graded_min: Optional[float] = None
    graded_max: Optional[float] = None
    centre_tol: Optional[float] = None
    end_tol: Optional[float] = None
    shape_ratio: Optional[float] = None
    step: Optional[float] = None
    n_points: int = 0
    n_graded: int = 0
    is_flat: bool = False
    value_min: Optional[float] = None
    value_max: Optional[float] = None

    def graded_pairs(self) -> List[Tuple[float, float]]:
        return [(a, t) for a, t in zip(self.angles, self.tolerances) if t is not None]

    def describe(self) -> str:
        if self.centre_tol is None:
            return f"{self.role}(no graded points)"
        return (f"{self.role} {self.graded_min:g}..{self.graded_max:g} "
                f"centre={self.centre_tol:.7g} end={self.end_tol:.7g} "
                f"shape={self.shape_ratio:.4g} n={self.n_graded}")


def _band_stats(xs: Sequence[float], tols: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Centre / end / shape / step / flatness for one (x, tolerance) series."""
    ordered = sorted(zip(xs, tols), key=lambda p: p[0])
    pairs = [(x, t) for x, t in ordered if t is not None and t > 0]
    out: Dict[str, Any] = {
        "graded_min": None, "graded_max": None, "centre_tol": None, "end_tol": None,
        "shape_ratio": None, "step": None, "n_points": len(xs), "n_graded": len(pairs),
        "is_flat": False, "graded_regions": 0, "span": None,
    }
    if not pairs:
        return out
    gmin, gmax = pairs[0][0], pairs[-1][0]
    mid = (gmin + gmax) / 2.0
    centre = min(pairs, key=lambda p: abs(p[0] - mid))[1]
    end = max(pairs[0][1], pairs[-1][1])
    tolvals = [t for _, t in pairs]
    steps = [round(b - a, 6) for a, b in zip([p[0] for p in pairs], [p[0] for p in pairs][1:])]
    step = Counter(steps).most_common(1)[0][0] if steps else None
    # contiguous graded regions, in position order (the arrays are not always sorted)
    regions, inside = 0, False
    for _x, t in ordered:
        if t is not None and t > 0:
            if not inside:
                regions += 1
                inside = True
        else:
            inside = False
    out.update({
        "graded_min": gmin, "graded_max": gmax,
        "centre_tol": centre, "end_tol": end,
        "shape_ratio": (end / centre) if centre else None,
        "step": step, "n_graded": len(pairs),
        "is_flat": (max(tolvals) / min(tolvals)) <= 1.001 if min(tolvals) else False,
        "graded_regions": regions,
        "span": (max(xs) - min(xs)) if len(xs) > 1 else None,
    })
    return out


def extract_tables_from_cells(
    source: str, object_name: str, cells: Dict[Tuple[int, int], Any], doc_text: str
) -> List[AtpTable]:
    """Turn one embedded sheet's cell map into zero or more normalised tables.

    Layout handling: the header row is the first row carrying a cell whose text
    contains 'tol'.  Each 'tol' column closes a *column group* (the x/value
    columns since the previous tol column).  N-up sheets -- 'Angle | V.R. | Tol.'
    repeated four times across the page -- are a continuation of the same series,
    so their groups are concatenated left to right.
    """
    if not cells:
        return []
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})

    hdr_row = None
    for r in rows:
        if any(_TOL_RE.search(_hdr(cells.get((r, c)))) for c in cols):
            hdr_row = r
            break
    if hdr_row is None:
        return []

    headers = {c: _hdr(cells.get((hdr_row, c))) for c in cols}
    tol_cols = [c for c in cols if _TOL_RE.search(headers.get(c, ""))]
    if not tol_cols:
        return []

    # unit hint from a "Linearity Table For: <n> <unit>" style row above the header
    unit_hint = ""
    for r in rows:
        if r >= hdr_row:
            break
        for c in cols:
            v = cells.get((r, c))
            if isinstance(v, str) and v.strip().lower() in ("degrees", "inches", "deg", "in"):
                unit_hint = v.strip()
    # groups: (x/value cols, tol col)
    groups: List[Tuple[List[int], int]] = []
    prev = -1
    for tc in tol_cols:
        members = [c for c in cols if prev < c < tc]
        groups.append((members, tc))
        prev = tc
    groups = [g for g in groups if g[0]]
    if not groups:
        return []

    data_rows = [r for r in rows if r > hdr_row]
    series: List[Tuple[float, Optional[float], Optional[float]]] = []
    angle_hdr = value_hdr = tol_hdr = ""
    for members, tc in groups:
        # which member column is the x axis, which is the output value?
        named_x = [c for c in members if _ANGLE_RE.search(headers.get(c, ""))
                   and not _VALUE_RE.search(headers.get(c, ""))]
        named_v = [c for c in members if _VALUE_RE.search(headers.get(c, ""))]
        numeric = [c for c in members if not _IGNORE_RE.search(headers.get(c, ""))]
        if named_x:
            x_col = named_x[0]
        elif numeric:
            x_col = numeric[0]
        else:
            continue
        if named_v:
            v_col = named_v[-1]
        else:
            later = [c for c in numeric if c > x_col]
            v_col = later[-1] if later else None
        if not angle_hdr:
            angle_hdr = headers.get(x_col, "")
            value_hdr = headers.get(v_col, "") if v_col is not None else ""
            tol_hdr = headers.get(tc, "")
        for r in data_rows:
            x = _num(cells.get((r, x_col)))
            if x is None:
                continue
            tol = parse_tolerance(cells.get((r, tc)))
            val = _num(cells.get((r, v_col))) if v_col is not None else None
            series.append((x, tol, val))

    if len(series) < 3:
        return []
    # N-up groups repeat the header spacing; de-duplicate identical x entries.
    seen: Dict[float, Tuple[Optional[float], Optional[float]]] = {}
    order: List[float] = []
    for x, tol, val in series:
        key = round(x, 6)
        if key not in seen:
            seen[key] = (tol, val)
            order.append(key)
        elif seen[key][0] is None and tol is not None:
            seen[key] = (tol, val)
    order.sort()
    angles = order
    tols = [seen[x][0] for x in order]
    vals = [seen[x][1] for x in order]

    # Validity gate.  A "Tol." column also appears in the ATP's parameter list
    # ("Linearity | --- | 0.04 | Percent"), and in a workbook like
    # 6607-Final Test Data Rev K.xlsx that list is the only thing with a tol
    # header.  Treating it as a per-point band produced a one-point "table" that
    # then fabricated RANGE_MISMATCH and SHAPE_MISMATCH on the highest-volume
    # model in the plant.  A real per-point table sweeps the travel.
    graded = [t for t in tols if t is not None]
    if len(angles) < 8 or len(graded) < 5:
        return []
    if len(set(angles)) < 8:
        return []

    t = AtpTable(
        source=source, object_name=object_name,
        angle_header=angle_hdr, value_header=value_hdr, tol_header=tol_hdr,
        unit_hint=unit_hint, angles=angles, tolerances=tols, values=vals,
    )
    stats = _band_stats(angles, tols)
    for k, v in stats.items():
        if hasattr(t, k):
            setattr(t, k, v)
    t.n_points = len(angles)
    numeric_vals = [v for v in vals if v is not None]
    if numeric_vals:
        t.value_min, t.value_max = min(numeric_vals), max(numeric_vals)
    _guess_role(t, doc_text)
    return [t]


def _guess_role(t: AtpTable, doc_text: str) -> None:
    """Role from the column header first, the surrounding prose second."""
    txt = (doc_text or "").lower()
    ev: List[str] = []
    role = "unknown"
    conf = "low"

    vh = t.value_header
    if _VALUE_RE.search(vh) and ("v.r" in vh or "vr" in vh.replace(".", "") or "ratio" in vh):
        role, conf = "trim", "high"
        ev.append(f"value column header {vh!r} is a voltage ratio (trim scale)")
    elif "volt" in vh:
        vmax = t.value_max if t.value_max is not None else 0.0
        if vmax > 1.5:
            role, conf = "final_test", "high"
            ev.append(f"value column header {vh!r} spans 0..{vmax:g} V (supply-voltage scale)")
        else:
            role, conf = "trim", "low"
            ev.append(f"value column header {vh!r} but only spans 0..{vmax:g}")

    has_trim_text = bool(re.search(r"linearity\s+trim\s+table|trim\s+data", txt))
    m_supply = re.search(r"supply\s+voltage\s+of\s+([\d.]+)\s*vdc", txt)
    if has_trim_text:
        ev.append("document text says LINEARITY TRIM TABLE / TRIM DATA")
    if m_supply:
        ev.append(f"document text names a supply voltage of {m_supply.group(1)} VDC")

    if role == "unknown":
        if has_trim_text and not m_supply:
            role, conf = "trim", "high" if t.value_max and t.value_max <= 1.5 else "low"
        elif m_supply and not has_trim_text:
            role, conf = "final_test", "low"
        elif t.value_max is not None:
            role = "trim" if t.value_max <= 1.5 else "final_test"
            conf = "low"
            ev.append(f"fallback: value column spans 0..{t.value_max:g}")

    # Corroboration: both section headings present and both scales present -> high.
    if has_trim_text and m_supply:
        try:
            supply = float(m_supply.group(1))
        except ValueError:
            supply = None
        if role == "final_test" and supply and t.value_max and abs(t.value_max - supply) < 0.2:
            conf = "high"
            ev.append(f"value column max {t.value_max:g} matches the stated supply {supply:g} VDC")
        elif role == "trim" and t.value_max and t.value_max <= 1.5:
            conf = "high"

    t.role = role
    t.role_confidence = conf
    t.role_evidence = "; ".join(ev) if ev else "no header or prose evidence"


# =========================================================================== #
# 3. ATP index (walk the library, pick revisions, pull prose specs)
# =========================================================================== #

_REV_RE = re.compile(r"\brev\.?\s*\(?\s*([A-Z]{1,2}\d?|\d{1,3}|-)\s*\)?", re.I)


def parse_revision(text: str) -> Tuple[Optional[str], Tuple[int, int]]:
    """'Rev M' -> ('M', rank).  Letters rank below numbers, '-' (release) lowest."""
    m = _REV_RE.search(text or "")
    if not m:
        return None, (-1, 0)
    raw = m.group(1).upper()
    if raw == "-":
        return "-", (0, 0)
    if raw.isdigit():
        return raw, (2, int(raw))
    lm = re.match(r"^([A-Z]{1,2})(\d?)$", raw)
    if lm:
        letters, tail = lm.group(1), lm.group(2)
        rank = 0
        for ch in letters:
            rank = rank * 26 + (ord(ch) - 64)
        return raw, (1, rank * 10 + (int(tail) if tail else 0))
    return raw, (0, 0)


_PREFIX_RE = re.compile(r"^\d{0,2}[A-Z]{1,4}(?=\d)")


def normalise_model(raw: str) -> Optional[str]:
    """Canonical form for matching.  Strict: a wrong match is worse than none."""
    if not raw:
        return None
    s = str(raw).upper().strip()
    s = s.split("S/N")[0]
    s = re.split(r"[\t\n\r]", s)[0]
    s = s.replace(" ", " ").strip()
    s = re.sub(r"^(MODEL|MDL)\s*[:.\-]?\s*", "", s)
    # first whitespace-delimited token that starts with a digit or a known prefix
    tokens = [tk for tk in re.split(r"\s+", s) if tk]
    cand = None
    for tk in tokens:
        tk = tk.strip(".,;:()[]•*")
        if not tk:
            continue
        stripped = _PREFIX_RE.sub("", tk)
        if re.match(r"^\d", stripped):
            cand = stripped
            break
    if cand is None:
        return None
    cand = cand.strip(".,;:()[]-_")
    cand = re.sub(r"[^0-9A-Z\-]", "", cand)
    if not re.match(r"^\d{3,8}(-[0-9A-Z]{1,4})*[A-Z]?$", cand):
        return None
    return cand


_FILENAME_MODEL_RE = re.compile(
    r"^(?:ATP[-\s_]*)?(?:DS[-\s_]*)?(\d{3,8}[A-Z]?(?:-\d{1,3}[A-Z]?)?)")


def model_from_filename(stem: str) -> Optional[str]:
    """Model number out of a filename, used only when the document has no MODEL line.

    Stricter than :func:`normalise_model` on the dash suffix: it accepts only a
    *numeric* one, so '6607-Final Test Data Rev K' gives 6607 rather than the
    nonexistent model 6607-FINAL.  Also tolerates the 'ATP-8877-4-DS' naming.
    """
    s = _PREFIX_RE.sub("", str(stem).upper().strip())
    m = _FILENAME_MODEL_RE.match(s)
    if not m:
        return normalise_model(stem)
    cand = m.group(1)
    return cand if re.match(r"^\d{3,8}", cand) else None


def near_miss_key(model: str) -> str:
    """Loose key, used only to surface near-misses for a human -- never to match."""
    s = model.upper()
    s = re.sub(r"-0+(\d)", r"-\1", s)           # -007 -> -7
    s = re.sub(r"[A-Z]+$", "", s)               # 8506A -> 8506
    s = s.split("-")[0] if "-" in s and not re.match(r"^\d+-\d+$", s) else s
    return s


PROSE_PATTERNS = {
    "resistance": re.compile(
        r"resistance\s*:?\s*([\d,]+(?:\.\d+)?)\s*(?:ohms?\s*)?(?:±|\+/-|\+)\s*([\d,]+(?:\.\d+)?)\s*(%|ohms?|Ω)?",
        re.I),
    "electrical_angle": re.compile(
        r"electrical\s+angle\s*:?\s*(?:±|\+/-|\+)?\s*([\d.]+)\s*(?:°|o\b|deg)?\s*"
        r"(?:(?:±|\+/-|\+)\s*([\d.]+)\s*(%|°|deg)?)?",
        re.I),
    "electrical_angle_range": re.compile(
        r"electrical\s+angle\s*:?\s*([\d.]+)\s*°?\s*(?:-|to)\s*([\d.]+)\s*°", re.I),
    "linearity": re.compile(r"linearity\s*(?:\(([^)]+)\))?\s*:?\s*([^\n\r]{0,60})", re.I),
    "smoothness": re.compile(r"(?:output\s+)?smoothness\s*:?\s*([^\n\r]{0,40})", re.I),
    "atp_number": re.compile(r"\b(ATP[-\s]?[0-9A-Z\-]{2,20})", re.I),
    # the colon is required and the label must start a line: without that,
    # 'customer' anywhere in a revision note steals the field
    "customer": re.compile(r"^\s*customer\s*(?:no\.?)?\s*:\s*([^\n\r\t]{0,60})", re.I | re.M),
    "drawing": re.compile(r"^\s*drawing\s*(?:no\.?)?\s*:\s*([^\n\r\t]{0,60})", re.I | re.M),
    "spec": re.compile(r"^\s*spec\.?\s*:\s*([^\n\r\t]{0,60})", re.I | re.M),
}


@dataclass
class AtpSheet:
    path: str
    ext: str
    filename_model: Optional[str] = None
    text_models: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)      # normalised, deduped
    revision: Optional[str] = None
    revision_rank: Tuple[int, int] = (-1, 0)
    atp_number: Optional[str] = None
    customer: Optional[str] = None
    drawing: Optional[str] = None
    spec: Optional[str] = None
    resistance_nominal: Optional[float] = None
    resistance_tol: Optional[float] = None
    resistance_tol_unit: Optional[str] = None
    electrical_angle: Optional[float] = None
    electrical_angle_tol: Optional[float] = None
    electrical_angle_tol_unit: Optional[str] = None
    linearity_type: Optional[str] = None
    linearity_text: Optional[str] = None
    smoothness: Optional[str] = None
    tables: List[AtpTable] = field(default_factory=list)
    read_status: str = "ok"          # ok | no_text | unreadable | pdf_no_text | skipped


def _clean_field(s: Optional[str]) -> Optional[str]:
    """Trim one prose value.  textutil renders Word table cell breaks as \\x07,
    so everything after the first control character belongs to the next cell."""
    if not s:
        return None
    s = re.split(r"[\x00-\x08\x0b-\x1f]", s)[0]
    s = re.sub(r"\s+", " ", s).strip(" \t:-")
    return s or None


def parse_sheet(path: Path, root: Path) -> AtpSheet:
    ext = path.suffix.lower()
    sheet = AtpSheet(path=str(path), ext=ext)
    rev, rank = parse_revision(path.name)
    sheet.revision, sheet.revision_rank = rev, rank

    fm = model_from_filename(path.stem)
    sheet.filename_model = fm

    text = ""
    if ext in DOC_EXT:
        text = textutil_text(path)
    elif ext in DOCX_EXT:
        # both, because each misses things the other catches: textutil drops
        # some sheets entirely, and it renders table cells more faithfully
        text = docx_text_from_zip(path) + "\n" + textutil_text(path)
    elif ext in PDF_EXT:
        text = pdf_text(path)
        if not text.strip():
            sheet.read_status = "pdf_no_text"
    elif ext in XLSX_EXT and ext != ".xls":
        text = xlsx_text(path)

    if text:
        for line in text.splitlines():
            if re.match(r"\s*MODEL\s*[:.]", line, re.I):
                nm = normalise_model(line)
                if nm and nm not in sheet.text_models:
                    sheet.text_models.append(nm)
        m = PROSE_PATTERNS["atp_number"].search(text)
        if m:
            sheet.atp_number = _clean_field(m.group(1)).upper().replace(" ", "-")
        for key, attr in (("customer", "customer"), ("drawing", "drawing"), ("spec", "spec")):
            m = PROSE_PATTERNS[key].search(text)
            if m:
                setattr(sheet, attr, _clean_field(m.group(1)))
        m = PROSE_PATTERNS["resistance"].search(text)
        if m:
            try:
                sheet.resistance_nominal = float(m.group(1).replace(",", ""))
                sheet.resistance_tol = float(m.group(2).replace(",", ""))
                sheet.resistance_tol_unit = "%" if (m.group(3) or "").strip() == "%" else "ohms"
            except ValueError:
                pass
        m = PROSE_PATTERNS["electrical_angle_range"].search(text)
        if m:
            try:
                lo, hi = float(m.group(1)), float(m.group(2))
                sheet.electrical_angle = (lo + hi) / 2.0
                sheet.electrical_angle_tol = (hi - lo) / 2.0
                sheet.electrical_angle_tol_unit = "deg"
            except ValueError:
                pass
        else:
            m = PROSE_PATTERNS["electrical_angle"].search(text)
            if m:
                try:
                    sheet.electrical_angle = float(m.group(1))
                    if m.group(2):
                        sheet.electrical_angle_tol = float(m.group(2))
                        sheet.electrical_angle_tol_unit = (m.group(3) or "deg").strip() or "deg"
                except ValueError:
                    pass
        m = re.search(r"linearity\s*(\([^)]*\))?\s*:\s*([^\n\r\t]{0,50})", text, re.I)
        if m:
            paren = (m.group(1) or "").strip("() ")
            body = _clean_field(m.group(2)) or ""
            sheet.linearity_text = body
            kind = paren or body
            kl = kind.lower()
            if "absolute" in kl:
                sheet.linearity_type = "Absolute"
            elif "independ" in kl:
                sheet.linearity_type = "Independent"
            elif "zero" in kl and "based" in kl:
                sheet.linearity_type = "Zero-Based"
            elif "terminal" in kl:
                sheet.linearity_type = "Terminal-Based"
            else:
                sheet.linearity_type = kind[:30] or None
        m = PROSE_PATTERNS["smoothness"].search(text)
        if m:
            sheet.smoothness = _clean_field(m.group(1))

    # tables
    try:
        raw_tables: List[Tuple[str, Dict[Tuple[int, int], Any]]] = []
        if ext in DOC_EXT:
            raw_tables = doc_embedded_tables(path)
        elif ext in DOCX_EXT:
            raw_tables = docx_embedded_tables(path)
        elif ext in XLSX_EXT and ext != ".xls":
            raw_tables = xlsx_tables(path)
        for obj, cells in raw_tables:
            sheet.tables.extend(extract_tables_from_cells(str(path), obj, cells, text))
    except Exception as exc:
        sheet.read_status = "unreadable"
        log.debug("tables %s: %s", path, exc)

    models: List[str] = []
    for m in sheet.text_models:
        if m not in models:
            models.append(m)
    if fm and fm not in models:
        models.append(fm)
    sheet.models = models
    if not text and ext in (DOC_EXT | DOCX_EXT):
        sheet.read_status = "no_text"
    return sheet


def walk_library(root: Path) -> Tuple[List[Path], Counter, List[str]]:
    """All candidate ATP files under root, plus a skip tally."""
    files, skipped = [], Counter()
    skipped_names: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if SKIP_DIR_RE.search("/" + rel + "/"):
            skipped["in_archive_dir"] += len(filenames)
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not SKIP_DIR_RE.search("/" + d + "/")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if SKIP_NAME_RE.search(fn):
                skipped["lock_obsolete_or_nondoc"] += 1
                skipped_names.append(str(p))
                continue
            ext = p.suffix.lower()
            if ext in DOC_EXT or ext in DOCX_EXT or ext in XLSX_EXT or ext in PDF_EXT:
                files.append(p)
            else:
                skipped["other_extension"] += 1
    return sorted(files), skipped, skipped_names


@dataclass
class AtpModelEntry:
    model: str
    prose_sheet: Optional[AtpSheet] = None
    table_sheet: Optional[AtpSheet] = None
    tables: List[AtpTable] = field(default_factory=list)
    all_sheets: List[str] = field(default_factory=list)
    revision_tie: bool = False


def build_atp_index(root: Path, limit: Optional[int] = None) -> Tuple[Dict[str, AtpModelEntry],
                                                                     List[AtpSheet], Counter, List[str]]:
    files, skipped, skipped_names = walk_library(root)
    if limit:
        files = files[:limit]
    sheets: List[AtpSheet] = []
    for i, p in enumerate(files):
        if i and i % 250 == 0:
            log.info("  ... %d/%d ATP files read", i, len(files))
        try:
            sheets.append(parse_sheet(p, root))
        except Exception as exc:
            s = AtpSheet(path=str(p), ext=p.suffix.lower(), read_status="unreadable")
            log.debug("parse %s: %s", p, exc)
            sheets.append(s)

    by_model: Dict[str, List[AtpSheet]] = defaultdict(list)
    for s in sheets:
        for m in s.models:
            by_model[m].append(s)

    index: Dict[str, AtpModelEntry] = {}
    for model, group in by_model.items():
        group = sorted(group, key=lambda s: (s.revision_rank, s.path), reverse=True)
        entry = AtpModelEntry(model=model, all_sheets=[s.path for s in group])
        top_rank = group[0].revision_rank
        entry.revision_tie = sum(1 for s in group if s.revision_rank == top_rank) > 1
        for s in group:
            if entry.prose_sheet is None and (s.resistance_nominal is not None
                                              or s.electrical_angle is not None
                                              or s.linearity_type):
                entry.prose_sheet = s
            if entry.table_sheet is None and s.tables:
                entry.table_sheet = s
                entry.tables = s.tables
        if entry.prose_sheet is None:
            entry.prose_sheet = group[0]
        index[model] = entry
    return index, sheets, skipped, skipped_names


# =========================================================================== #
# 4. Station-side: what the limit columns in the ingested files actually say
# =========================================================================== #

@dataclass
class StationSignature:
    model: str
    station: str                       # 'trim' | 'final_test'
    n_tracks: int = 0
    n_tracks_all_time: int = 0
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_min_all: Optional[str] = None
    date_max_all: Optional[str] = None
    positions: List[float] = field(default_factory=list)
    limits: List[Optional[float]] = field(default_factory=list)
    graded_min: Optional[float] = None
    graded_max: Optional[float] = None
    centre_tol: Optional[float] = None
    end_tol: Optional[float] = None
    shape_ratio: Optional[float] = None
    step: Optional[float] = None
    n_points: int = 0
    n_graded: int = 0
    is_flat: bool = False
    graded_regions: int = 0
    raw_variants: List[Dict[str, Any]] = field(default_factory=list)

    def describe(self) -> str:
        if self.centre_tol is None:
            return "no graded points"
        return (f"{self.graded_min:g}..{self.graded_max:g} centre={self.centre_tol:.7g} "
                f"end={self.end_tol:.7g} shape={self.shape_ratio:.4g} n={self.n_points}")


def _json_floats(raw: Any, absolute: bool) -> Optional[List[Optional[float]]]:
    """A stored JSON array -> floats, with NaN/non-numeric folded to None.

    ``absolute=True`` for tolerance columns (the band is symmetric and some
    stations store the lower limit as a negative).  Positions keep their sign --
    taking abs() there silently folds a -28..+28 sweep into 0..28.
    """
    if raw is None:
        return None
    try:
        arr = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    out: List[Optional[float]] = []
    for v in arr:
        if isinstance(v, bool) or v is None:
            out.append(None)
        elif isinstance(v, (int, float)):
            f = float(v)
            out.append(None if math.isnan(f) or math.isinf(f) else (abs(f) if absolute else f))
        else:
            out.append(None)
    return out


STATION_QUERIES = {
    "trim": """
        SELECT a.model, t.position_data, t.upper_limits, a.file_date
        FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id
        WHERE t.upper_limits IS NOT NULL AND t.position_data IS NOT NULL
          AND (t.status IS NULL OR t.status <> 'UNTRIMMED')
    """,
    "final_test": """
        SELECT f.model, t.position_data, t.upper_limits, f.file_date
        FROM final_test_tracks t JOIN final_test_results f ON f.id = t.final_test_id
        WHERE t.upper_limits IS NOT NULL AND t.position_data IS NOT NULL
    """,
}


def collect_station_signatures(
    con: sqlite3.Connection, since: str, models: Optional[Sequence[str]] = None
) -> Dict[Tuple[str, str], List[StationSignature]]:
    """Group every track's limit array into physical *band* signatures.

    Two levels, on purpose:

    * ``raw_sig``  -- the limit array itself, rounded to 6 dp (the literal
      "signature of their limit array").
    * ``band_sig`` -- centre / end / graded range.  The same physical band is
      often sampled on two different position grids (57 points at 1 deg and 111
      points at 0.5 deg, say); those are one band, not two specs, so the report
      groups on the band and lists the raw variants inside it.

    Rounding the *graded range* to 0.1 also absorbs the float noise a few
    stations write into the position column (-27.9835 instead of -28.0); the
    limit values themselves are byte-identical in those rows.
    """
    want = set(models) if models else None
    acc: Dict[Tuple[str, str], Dict[Tuple, Dict[str, Any]]] = defaultdict(dict)

    for station, sql in STATION_QUERIES.items():
        cur = con.cursor()
        cur.execute(sql)
        while True:
            chunk = cur.fetchmany(2000)
            if not chunk:
                break
            for model, pos_s, lim_s, fdate in chunk:
                if not model or (want is not None and model not in want):
                    continue
                lims = _json_floats(lim_s, absolute=True)
                poss = _json_floats(pos_s, absolute=False)
                if lims is None or poss is None or len(lims) != len(poss):
                    continue
                if any(p is None for p in poss):
                    continue
                stats = _band_stats(poss, lims)
                if stats["centre_tol"] is None:
                    continue
                # The graded window goes into the key as a *fraction of the swept
                # span*, not as raw units: the position axis is degrees for some
                # models and inches of travel for others, and several stations
                # write float noise into it (-27.9835 where the next unit says
                # -28.0).  1% of span absorbs that noise without ever merging a
                # +/-22 window with a +/-28 one.
                span = stats.get("span") or 1.0
                band_key = (
                    round(stats["centre_tol"], 6), round(stats["end_tol"], 6),
                    round(stats["graded_min"] / span, 2), round(stats["graded_max"] / span, 2),
                    stats["graded_regions"],
                )
                raw_key = tuple(None if v is None else round(v, 6) for v in lims)
                slot = acc[(model, station)].setdefault(band_key, {
                    "n_recent": 0, "n_all": 0,
                    "dmin": None, "dmax": None, "dmin_all": None, "dmax_all": None,
                    "raw": {},
                })
                d = (fdate or "")[:19]
                recent = d >= since
                rv = slot["raw"].get(raw_key)
                if rv is None:
                    rv = slot["raw"][raw_key] = {
                        "n": 0, "n_all": 0, "pts": len(lims), "dmin": None, "dmax": None,
                        "pos": poss, "lim": lims, "stats": stats,
                    }
                slot["n_all"] += 1
                slot["dmin_all"] = d if slot["dmin_all"] is None else min(slot["dmin_all"], d)
                slot["dmax_all"] = d if slot["dmax_all"] is None else max(slot["dmax_all"], d)
                rv["n_all"] += 1
                rv["dmin"] = d if rv["dmin"] is None else min(rv["dmin"], d)
                rv["dmax"] = d if rv["dmax"] is None else max(rv["dmax"], d)
                if recent:
                    rv["n"] += 1
                    slot["n_recent"] += 1
                    slot["dmin"] = d if slot["dmin"] is None else min(slot["dmin"], d)
                    slot["dmax"] = d if slot["dmax"] is None else max(slot["dmax"], d)
        cur.close()

    out: Dict[Tuple[str, str], List[StationSignature]] = {}
    for key, bands in acc.items():
        model, station = key
        sigs: List[StationSignature] = []
        for band_key, slot in bands.items():
            # Deterministic representative: the raw array variant that the most
            # recent-window tracks were graded against (then all-time, then the
            # finest grid) -- never "whichever row the cursor happened to end on".
            rep = max(slot["raw"].values(),
                      key=lambda v: (v["n"], v["n_all"], v["pts"]))
            st = rep["stats"]
            sig = StationSignature(
                model=model, station=station,
                n_tracks=slot["n_recent"], n_tracks_all_time=slot["n_all"],
                date_min=slot["dmin"], date_max=slot["dmax"],
                date_min_all=slot["dmin_all"], date_max_all=slot["dmax_all"],
                positions=rep["pos"], limits=rep["lim"],
                graded_min=st["graded_min"], graded_max=st["graded_max"],
                centre_tol=st["centre_tol"], end_tol=st["end_tol"],
                shape_ratio=st["shape_ratio"], step=st["step"],
                n_points=len(rep["pos"]), n_graded=st["n_graded"],
                is_flat=st["is_flat"], graded_regions=st["graded_regions"],
            )
            sig.raw_variants = sorted(
                ({"n_tracks": v["n"], "n_tracks_all_time": v["n_all"], "n_points": v["pts"],
                  "date_min": v["dmin"], "date_max": v["dmax"]}
                 for v in slot["raw"].values()),
                key=lambda d: (-d["n_tracks"], -d["n_tracks_all_time"]))
            sigs.append(sig)
        sigs.sort(key=lambda s: (-s.n_tracks, -s.n_tracks_all_time))
        out[key] = sigs
    return out


def recent_track_counts(con: sqlite3.Connection, since: str) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"trim": 0, "final_test": 0})
    cur = con.cursor()
    cur.execute("""SELECT a.model, COUNT(*) FROM track_results t
                   JOIN analysis_results a ON a.id = t.analysis_id
                   WHERE a.file_date >= ? AND (t.status IS NULL OR t.status <> 'UNTRIMMED')
                   GROUP BY a.model""", (since,))
    for m, n in cur.fetchall():
        if m:
            counts[m]["trim"] = n
    cur.execute("""SELECT f.model, COUNT(*) FROM final_test_tracks t
                   JOIN final_test_results f ON f.id = t.final_test_id
                   WHERE f.file_date >= ? GROUP BY f.model""", (since,))
    for m, n in cur.fetchall():
        if m:
            counts[m]["final_test"] = n
    cur.close()
    return counts


# =========================================================================== #
# 5. Axis alignment + comparison
# =========================================================================== #

@dataclass
class AxisAlignment:
    kind: str                 # 'as_is' | 'centred' | 'zero_based' | 'rescaled' | 'none'
    offset: float = 0.0
    scale: float = 1.0
    span_error: float = 0.0
    confidence: str = "low"

    def apply(self, x: float) -> float:
        return (x + self.offset) * self.scale


def align_axis(atp_angles: Sequence[float], station_pos: Sequence[float]) -> AxisAlignment:
    """Map the ATP's x column onto the station's position axis.

    ATP sheets print the x axis three different ways (centred 'Test Angle',
    0-based 'Angle', and 'Travel' in inches) and the stations are equally
    inconsistent, so try the offset-only transforms first -- those preserve the
    physical unit and let the range comparison quote real angles.  Only fall back
    to an affine rescale (different units entirely) as a last resort, and say so.
    """
    if not atp_angles or not station_pos:
        return AxisAlignment("none", confidence="low")
    a_lo, a_hi = min(atp_angles), max(atp_angles)
    s_lo, s_hi = min(station_pos), max(station_pos)
    a_span, s_span = a_hi - a_lo, s_hi - s_lo
    if a_span <= 0 or s_span <= 0:
        return AxisAlignment("none", confidence="low")

    span_err = abs(a_span - s_span) / s_span
    candidates: List[AxisAlignment] = []
    if span_err <= AXIS_SPAN_TOL:
        a_mid, s_mid = (a_lo + a_hi) / 2.0, (s_lo + s_hi) / 2.0
        centre_err = abs(a_mid - s_mid) / s_span
        candidates.append(AxisAlignment("as_is", 0.0, 1.0, span_err,
                                        "high" if centre_err <= 0.02 else "low"))
        candidates.append(AxisAlignment("centred", s_mid - a_mid, 1.0, span_err, "high"))
        candidates.append(AxisAlignment("zero_based", s_lo - a_lo, 1.0, span_err, "high"))
        # Score by the WORST end, not the sum.  For pure offsets the sum of the
        # two end errors is constant, so summing leaves all three tied and lets
        # float noise choose -- and it would happily pick the shift that pushes
        # one end right off the station's travel.  Minimax picks the centred fit,
        # which is what a symmetric bowtie actually means.  Order breaks ties, so
        # 'as_is' wins when the sheet already uses the station's own axis.
        best = None
        for c in candidates:
            err = max(abs(c.apply(a_lo) - s_lo), abs(c.apply(a_hi) - s_hi)) / s_span
            if best is None or err < best[0] - 1e-12:
                best = (err, c)
        if best:
            best[1].confidence = "high" if best[0] <= 0.02 else "low"
            return best[1]
    # different units: affine map of the full ATP span onto the full station span
    return AxisAlignment("rescaled", -a_lo + (s_lo * a_span / s_span), s_span / a_span,
                         span_err, "low")


def _interp(xs: Sequence[float], ys: Sequence[Optional[float]], x: float) -> Optional[float]:
    """Linear interpolation of a station band at x; None outside the graded span."""
    pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    if x < pts[0][0] - ANGLE_TOL or x > pts[-1][0] + ANGLE_TOL:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 - ANGLE_TOL <= x <= x1 + ANGLE_TOL:
            if abs(x1 - x0) < 1e-12:
                return y0
            f = (x - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    return None


@dataclass
class Finding:
    code: str
    severity: str             # 'high' | 'medium' | 'low'
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


def _table_label(t: AtpTable) -> str:
    """Short, unambiguous name for one ATP table -- several sheets carry three
    tables that all guess the same role, so 'the trim table' is not enough."""
    bits = [f"role={t.role}"]
    if t.value_header:
        bits.append(f"`{t.value_header}`")
    if t.value_max is not None:
        bits.append(f"0..{t.value_max:g}")
    if t.shape_ratio:
        bits.append(f"centre {t.centre_tol:.6g} end {t.end_tol:.6g} shape {t.shape_ratio:.4g}")
    bits.append(f"[{t.object_name}]")
    return " ".join(bits)


def _is_monotonic(xs: Sequence[float], slack: float = 0.02) -> bool:
    """Does this position column actually sweep?  A few reversals are tolerated
    (encoder jitter); a column that wanders is not a travel axis at all."""
    if len(xs) < 3:
        return True
    ups = sum(1 for a, b in zip(xs, xs[1:]) if b > a)
    downs = sum(1 for a, b in zip(xs, xs[1:]) if b < a)
    n = len(xs) - 1
    return max(ups, downs) >= n * (1 - slack)


def _shape_check(sig: "StationSignature", table: AtpTable,
                 other_tables: Sequence[AtpTable], info: Dict[str, Any]) -> List[Finding]:
    """End/centre ratio -- the one comparison that needs no axis alignment,
    so it stands even when the two position columns are incommensurable."""
    out: List[Finding] = []
    if not (table.shape_ratio and sig.shape_ratio):
        return out
    rel = abs(sig.shape_ratio - table.shape_ratio) / table.shape_ratio
    info["shape_rel_diff"] = rel
    if rel <= SHAPE_TOL:
        return out
    swap = None
    for other in other_tables:
        if other is table or not other.shape_ratio:
            continue
        if abs(sig.shape_ratio - other.shape_ratio) / other.shape_ratio <= SHAPE_TOL:
            swap = other
            break
    if swap is not None:
        out.append(Finding(
            "TABLE_SWAP", "high",
            f"station shape {sig.shape_ratio:.4g} does not match the ATP table selected for "
            f"this station ({_table_label(table)}) but does match another table in the same "
            f"ATP ({_table_label(swap)}) -- the station appears to be grading to the wrong "
            f"ATP table",
            {"station_shape": sig.shape_ratio, "this_table_shape": table.shape_ratio,
             "matched_table_role": swap.role, "matched_table_shape": swap.shape_ratio,
             "matched_table_object": swap.object_name,
             "matched_table_source": swap.source,
             "this_table_object": table.object_name}))
    else:
        out.append(Finding(
            "SHAPE_MISMATCH", "high",
            f"band shape (end/centre) differs by {rel * 100:.1f}%: ATP "
            f"{table.shape_ratio:.4g} vs station {sig.shape_ratio:.4g}",
            {"atp_shape": table.shape_ratio, "station_shape": sig.shape_ratio,
             "rel_diff": rel}))
    return out


def compare(sig: StationSignature, table: AtpTable, other_tables: Sequence[AtpTable]
            ) -> Tuple[List[Finding], Dict[str, Any]]:
    """All checks for one (station signature, ATP table) pair."""
    findings: List[Finding] = []
    align = align_axis(table.angles, sig.positions)
    monotonic = _is_monotonic(sig.positions)
    info: Dict[str, Any] = {
        "axis_kind": align.kind, "axis_offset": align.offset, "axis_scale": align.scale,
        "axis_span_error": align.span_error, "axis_confidence": align.confidence,
        "station_axis_monotonic": monotonic,
    }

    # --- shape, first: it is the one check that needs no axis at all --------- #
    shape_findings = _shape_check(sig, table, other_tables, info)

    if not monotonic:
        findings.append(Finding(
            "STATION_AXIS_NOT_MONOTONIC", "high",
            f"the station's position column is not monotonic "
            f"({min(sig.positions):g}..{max(sig.positions):g} over {len(sig.positions)} points, "
            f"values jumping back and forth) -- it is not a travel axis, so no angle-by-angle "
            f"comparison is possible and the ingested per-point data for this model should be "
            f"re-checked"))
        return findings + shape_findings, info
    if align.kind == "none" or (align.kind == "rescaled" and align.span_error > AXIS_GIVE_UP):
        findings.append(Finding(
            "AXIS_UNRESOLVED", "medium",
            f"ATP x axis ({table.angle_header or '?'} {min(table.angles):g}.."
            f"{max(table.angles):g}{' ' + table.unit_hint if table.unit_hint else ''}) sweeps a "
            f"different quantity from the station axis "
            f"({min(sig.positions):g}..{max(sig.positions):g}); spans differ by "
            f"{align.span_error * 100:.0f}%, so no angle comparison was attempted "
            f"(the shape check above is axis-free and still stands)"))
        return findings + shape_findings, info
    if align.kind == "rescaled":
        findings.append(Finding(
            "AXIS_RESCALED", "low",
            f"ATP x axis ({table.angle_header or '?'} {min(table.angles):g}..{max(table.angles):g}"
            f"{' ' + table.unit_hint if table.unit_hint else ''}) and the station axis "
            f"({min(sig.positions):g}..{max(sig.positions):g}) use different units; compared "
            f"proportionally, so the exact angles below are approximate"))
    findings.extend(shape_findings)

    # --- range ------------------------------------------------------------- #
    atp_graded = [(align.apply(a), t) for a, t in zip(table.angles, table.tolerances)]
    atp_graded_angles = [a for a, t in atp_graded if t is not None]
    if not atp_graded_angles:
        findings.append(Finding("NO_GRADED_POINTS_IN_ATP", "medium",
                                "the ATP table has no numeric tolerance anywhere"))
        return findings, info
    a_lo, a_hi = min(atp_graded_angles), max(atp_graded_angles)
    s_lo, s_hi = sig.graded_min, sig.graded_max
    info.update({"atp_graded_min_on_station_axis": a_lo, "atp_graded_max_on_station_axis": a_hi})

    tol_band = max(ANGLE_TOL, 0.01 * (max(sig.positions) - min(sig.positions)))
    station_only = [p for p, l in zip(sig.positions, sig.limits)
                    if l is not None and (p < a_lo - tol_band or p > a_hi + tol_band)]
    atp_only = [a for a, t in atp_graded
                if t is not None and (a < s_lo - tol_band or a > s_hi + tol_band)]
    if station_only or atp_only:
        parts = []
        if station_only:
            parts.append(f"station grades {len(station_only)} point(s) the ATP marks not-graded "
                         f"({_fmt_angles(station_only)})")
        if atp_only:
            parts.append(f"ATP grades {len(atp_only)} point(s) the station does not "
                         f"({_fmt_angles(atp_only)})")
        findings.append(Finding(
            # unit-independent, so high confidence -- unless the axes had to be
            # scaled onto each other, in which case the angles are approximate
            "RANGE_MISMATCH", "medium" if align.kind == "rescaled" else "high",
            f"graded window differs: ATP {a_lo:g}..{a_hi:g} vs station {s_lo:g}..{s_hi:g}; "
            + "; ".join(parts),
            {"station_only_points": len(station_only), "atp_only_points": len(atp_only),
             "station_only_angles": [round(x, 4) for x in station_only[:40]],
             "atp_only_angles": [round(x, 4) for x in atp_only[:40]]}))

    # --- value ratio at matched angles ------------------------------------- #
    ratios: List[Tuple[float, float]] = []
    for a, t in atp_graded:
        if t is None or t <= 0:
            continue
        s = _interp(sig.positions, sig.limits, a)
        if s is None or s <= 0:
            continue
        ratios.append((a, s / t))
    if ratios:
        rvals = [r for _, r in ratios]
        rmin, rmax = min(rvals), max(rvals)
        rmean = sum(rvals) / len(rvals)
        info.update({"value_ratio_min": rmin, "value_ratio_max": rmax, "value_ratio_mean": rmean,
                     "value_ratio_points": len(rvals)})
        constant = (rmax - rmin) <= RATIO_TOL * max(1e-12, rmean)
        if constant:
            clean = next((c for c in CLEAN_SCALES if abs(rmean - c) <= RATIO_TOL * c), None)
            if clean is not None:
                findings.append(Finding(
                    "VALUE_RATIO_CLEAN", "low",
                    f"station tolerance is a constant x{clean:g} of the ATP table "
                    f"(mean {rmean:.5g} over {len(rvals)} angles) -- consistent with a "
                    f"V.R.-versus-supply-voltage / unit difference",
                    {"scale": clean, "mean": rmean}))
            else:
                findings.append(Finding(
                    "VALUE_RATIO_CONSTANT_UNEXPLAINED", "medium",
                    f"station tolerance is a constant x{rmean:.5g} of the ATP table "
                    f"({len(rvals)} angles, spread {rmax - rmin:.2e}) -- a clean scale, but not "
                    f"one of {', '.join(f'{c:g}' for c in CLEAN_SCALES)}",
                    {"mean": rmean}))
        else:
            findings.append(Finding(
                "VALUE_RATIO_VARIES", "high",
                f"station/ATP tolerance ratio is not constant: {rmin:.5g}..{rmax:.5g} "
                f"across {len(rvals)} matched angles (mean {rmean:.5g}) -- the two bands are "
                f"different curves, not the same curve in different units",
                {"min": rmin, "max": rmax, "mean": rmean,
                 "worst_angles": [round(a, 3) for a, r in
                                  sorted(ratios, key=lambda p: p[1])[:3]
                                  + sorted(ratios, key=lambda p: -p[1])[:3]]}))

    # --- interpolation ------------------------------------------------------ #
    atp_pts = sorted([(a, t) for a, t in atp_graded if t is not None])
    if len(atp_pts) >= 2 and len(sig.positions) > len(atp_pts):
        bad = []
        for p, l in zip(sig.positions, sig.limits):
            if l is None:
                continue
            if p < atp_pts[0][0] - ANGLE_TOL or p > atp_pts[-1][0] + ANGLE_TOL:
                continue
            if any(abs(p - a) <= ANGLE_TOL for a, _ in atp_pts):
                continue
            lo = max((x for x, _ in atp_pts if x <= p), default=None)
            hi = min((x for x, _ in atp_pts if x >= p), default=None)
            if lo is None or hi is None or hi == lo:
                continue
            s_lo_v = _interp(sig.positions, sig.limits, lo)
            s_hi_v = _interp(sig.positions, sig.limits, hi)
            if s_lo_v is None or s_hi_v is None:
                continue
            expect = s_lo_v + (p - lo) / (hi - lo) * (s_hi_v - s_lo_v)
            if abs(expect - l) > INTERP_TOL:
                bad.append((p, l, expect))
        info["interpolation_checked"] = True
        if bad:
            findings.append(Finding(
                "INTERPOLATION_MISMATCH", "medium",
                f"{len(bad)} station point(s) between ATP grid angles are not the linear "
                f"interpolation of their neighbours (worst "
                f"{max(abs(e - l) for _, l, e in bad):.3g} > {INTERP_TOL:g})",
                {"count": len(bad),
                 "examples": [{"pos": round(p, 4), "station": l, "expected": round(e, 9)}
                              for p, l, e in bad[:5]]}))
    return findings, info


def _fmt_angles(vals: Sequence[float], limit: int = 8) -> str:
    v = sorted(set(round(x, 3) for x in vals))
    if len(v) <= limit:
        return ", ".join(f"{x:g}" for x in v)
    return ", ".join(f"{x:g}" for x in v[:limit // 2]) + ", ... , " + \
        ", ".join(f"{x:g}" for x in v[-limit // 2:])


def pick_table(sig: StationSignature, tables: Sequence[AtpTable]
               ) -> Tuple[Optional[AtpTable], str]:
    """The ATP table this station *should* be grading to, and how it was chosen."""
    if not tables:
        return None, "none"
    want = sig.station
    byrole = [t for t in tables if t.role == want]
    if byrole:
        byrole.sort(key=lambda t: (t.role_confidence != "high", -t.n_graded))
        return byrole[0], "role_match"
    unknown = [t for t in tables if t.role == "unknown"]
    if unknown:
        return max(unknown, key=lambda t: t.n_graded), "role_unknown"
    # only the other station's table exists -- compare against it and let
    # TABLE_SWAP / SHAPE_MISMATCH say what happened
    return max(tables, key=lambda t: t.n_graded), "other_station_table"


# =========================================================================== #
# 6. Prose-spec cross-check against model_specs
# =========================================================================== #

def load_model_specs(con: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    cur = con.cursor()
    try:
        cur.execute("""SELECT model, linearity_type, total_resistance_min, total_resistance_max,
                              electrical_angle, electrical_angle_tol, electrical_angle_tol_type,
                              electrical_angle_unit, output_smoothness
                       FROM model_specs""")
    except sqlite3.Error:
        return {}
    out = {}
    for row in cur.fetchall():
        out[row[0]] = {
            "linearity_type": row[1], "total_resistance_min": row[2], "total_resistance_max": row[3],
            "electrical_angle": row[4], "electrical_angle_tol": row[5],
            "electrical_angle_tol_type": row[6], "electrical_angle_unit": row[7],
            "output_smoothness": row[8],
        }
    cur.close()
    return out


def check_prose(model: str, sheet: Optional[AtpSheet], spec: Optional[Dict[str, Any]]
                ) -> List[Finding]:
    findings: List[Finding] = []
    if sheet is None:
        return findings
    if spec is None:
        findings.append(Finding("SPEC_ROW_MISSING", "medium",
                                "the ATP carries prose specs but model_specs has no row for "
                                "this model"))
        return findings
    if sheet.resistance_nominal is not None and sheet.resistance_tol is not None:
        if sheet.resistance_tol_unit == "%":
            lo = sheet.resistance_nominal * (1 - sheet.resistance_tol / 100.0)
            hi = sheet.resistance_nominal * (1 + sheet.resistance_tol / 100.0)
        else:
            lo = sheet.resistance_nominal - sheet.resistance_tol
            hi = sheet.resistance_nominal + sheet.resistance_tol
        dlo, dhi = spec.get("total_resistance_min"), spec.get("total_resistance_max")
        if dlo is None or dhi is None:
            findings.append(Finding("SPEC_RESISTANCE_MISSING", "low",
                                    f"ATP says {lo:g}..{hi:g} ohms; model_specs has no range"))
        elif abs(dlo - lo) > max(1.0, 0.01 * abs(lo)) or abs(dhi - hi) > max(1.0, 0.01 * abs(hi)):
            findings.append(Finding(
                "SPEC_RESISTANCE_MISMATCH", "medium",
                f"ATP {lo:g}..{hi:g} ohms vs model_specs {dlo:g}..{dhi:g}",
                {"atp_min": lo, "atp_max": hi, "db_min": dlo, "db_max": dhi}))
    if sheet.electrical_angle is not None:
        dea = spec.get("electrical_angle")
        if dea is None:
            findings.append(Finding("SPEC_ANGLE_MISSING", "low",
                                    f"ATP says electrical angle {sheet.electrical_angle:g}; "
                                    f"model_specs has none"))
        elif abs(dea - sheet.electrical_angle) > max(0.05, 0.01 * abs(sheet.electrical_angle)):
            findings.append(Finding(
                "SPEC_ANGLE_MISMATCH", "medium",
                f"ATP electrical angle {sheet.electrical_angle:g} vs model_specs {dea:g}",
                {"atp": sheet.electrical_angle, "db": dea}))
        if sheet.electrical_angle_tol is not None and spec.get("electrical_angle_tol") is not None:
            dt = spec["electrical_angle_tol"]
            if abs(dt - sheet.electrical_angle_tol) > max(0.02, 0.02 * sheet.electrical_angle_tol):
                findings.append(Finding(
                    "SPEC_ANGLE_TOL_MISMATCH", "low",
                    f"ATP angle tolerance {sheet.electrical_angle_tol:g} vs model_specs {dt:g}"))
    if sheet.linearity_type and spec.get("linearity_type"):
        if sheet.linearity_type.lower()[:6] != str(spec["linearity_type"]).lower()[:6]:
            findings.append(Finding(
                "SPEC_LINEARITY_TYPE_MISMATCH", "medium",
                f"ATP linearity type {sheet.linearity_type!r} vs model_specs "
                f"{spec['linearity_type']!r}"))
    return findings


# =========================================================================== #
# 7. Audit driver
# =========================================================================== #

def months_ago(n: int) -> str:
    today = _dt.date.today()
    y, m = today.year, today.month - n
    while m <= 0:
        m += 12
        y -= 1
    return f"{y:04d}-{m:02d}-{min(today.day, 28):02d}"


def run_audit(db_path: Path, atp_root: Path, since: str, limit: Optional[int] = None
              ) -> Dict[str, Any]:
    log.info("indexing ATP library at %s", atp_root)
    index, sheets, skipped, skipped_names = build_atp_index(atp_root, limit=limit)
    log.info("  %d ATP files read, %d models identified", len(sheets), len(index))

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        log.info("reading station limit arrays (since %s)", since)
        sigs = collect_station_signatures(con, since)
        counts = recent_track_counts(con, since)
        specs = load_model_specs(con)
    finally:
        con.close()

    db_models = sorted({m for (m, _s) in sigs.keys()})
    atp_models = sorted(index.keys())
    db_norm = {}
    for m in db_models:
        n = normalise_model(m)
        db_norm[m] = n if n else None

    matched: Dict[str, str] = {}          # db model -> atp model key
    for m in db_models:
        n = db_norm[m]
        if n and n in index:
            matched[m] = n

    unmatched_db = [m for m in db_models if m not in matched]
    used_atp = set(matched.values())
    unmatched_atp = [m for m in atp_models if m not in used_atp]

    # Near misses, for a human to resolve -- never auto-matched.  Two rules:
    #   * same loose key (trailing letter / leading zeros in a dash suffix)
    #   * the ATP model ends with the DB model, i.e. a customer part-number
    #     prefix ('2001844205' in the library for DB model '1844205')
    atp_by_loose: Dict[str, List[str]] = defaultdict(list)
    for m in atp_models:
        atp_by_loose[near_miss_key(m)].append(m)
    near_misses = []
    for m in unmatched_db:
        n = db_norm[m] or m.upper()
        cands = set(atp_by_loose.get(near_miss_key(n), []))
        if len(n) >= 4:
            cands.update(a for a in atp_models
                         if a != n and a.endswith(n) and len(a) - len(n) <= 4
                         and a[:len(a) - len(n)].isdigit())
        if cands:
            near_misses.append({"db_model": m, "normalised": n,
                                "atp_candidates": sorted(cands)})

    results: List[Dict[str, Any]] = []
    for m in db_models:
        rec: Dict[str, Any] = {
            "model": m,
            "normalised": db_norm[m],
            "recent_tracks_trim": counts.get(m, {}).get("trim", 0),
            "recent_tracks_ft": counts.get(m, {}).get("final_test", 0),
            "atp_model_key": matched.get(m),
            "atp_sheets": [], "atp_revision": None, "atp_number": None, "customer": None,
            "atp_tables": [], "stations": {}, "prose_findings": [], "codes": [],
            "revision_tie": False,
        }
        rec["recent_tracks_total"] = rec["recent_tracks_trim"] + rec["recent_tracks_ft"]
        entry = index.get(matched[m]) if m in matched else None
        if entry is None:
            rec["codes"].append("NO_ATP")
        else:
            rec["atp_sheets"] = entry.all_sheets
            rec["revision_tie"] = entry.revision_tie
            ps = entry.prose_sheet
            if ps:
                rec["atp_revision"] = ps.revision
                rec["atp_number"] = ps.atp_number
                rec["customer"] = ps.customer
                rec["atp_prose"] = {
                    "path": ps.path, "drawing": ps.drawing, "spec": ps.spec,
                    "resistance_nominal": ps.resistance_nominal,
                    "resistance_tol": ps.resistance_tol,
                    "resistance_tol_unit": ps.resistance_tol_unit,
                    "electrical_angle": ps.electrical_angle,
                    "electrical_angle_tol": ps.electrical_angle_tol,
                    "electrical_angle_tol_unit": ps.electrical_angle_tol_unit,
                    "linearity_type": ps.linearity_type, "linearity_text": ps.linearity_text,
                    "smoothness": ps.smoothness,
                }
            rec["atp_tables"] = [{
                "source": t.source, "object": t.object_name, "role": t.role,
                "role_confidence": t.role_confidence, "role_evidence": t.role_evidence,
                "angle_header": t.angle_header, "value_header": t.value_header,
                "tol_header": t.tol_header, "unit_hint": t.unit_hint,
                "value_min": t.value_min, "value_max": t.value_max,
                "graded_min": t.graded_min, "graded_max": t.graded_max,
                "centre_tol": t.centre_tol, "end_tol": t.end_tol,
                "shape_ratio": t.shape_ratio, "step": t.step,
                "n_points": t.n_points, "n_graded": t.n_graded, "is_flat": t.is_flat,
            } for t in entry.tables]
            if not entry.tables:
                rec["codes"].append("NO_TABLE_IN_ATP")
            for f in check_prose(m, entry.prose_sheet, specs.get(m)):
                rec["prose_findings"].append(asdict(f))
                rec["codes"].append(f.code)

        for station in ("trim", "final_test"):
            slist = sigs.get((m, station), [])
            recent_only = [s for s in slist if s.n_tracks > 0]
            # a model that stopped running still gets audited, against its last
            # band -- but say so, so "n=0 recent tracks" is not read as a bug
            stale = not recent_only
            slist = recent_only or slist[:3]
            if not slist:
                rec["stations"][station] = {"status": "NO_STATION_DATA", "signatures": []}
                continue
            block: Dict[str, Any] = {"status": "stale" if stale else "ok",
                                     "signatures": []}
            if len(slist) > 1:
                # A band carrying a handful of tracks out of thousands is a
                # one-off fixture or a mis-set station, not "two specs in
                # production".  Both are reported; only the material case gets
                # the headline code, so the count means something.
                total = sum(s.n_tracks for s in slist) or 1
                material = [s for s in slist
                            if s.n_tracks >= 10 and s.n_tracks / total >= 0.05]
                rec["codes"].append("MULTIPLE_TABLES_IN_USE" if len(material) > 1
                                    else "MULTIPLE_TABLES_TAIL")
                block["multiple"] = [
                    {"n_tracks": s.n_tracks, "n_tracks_all_time": s.n_tracks_all_time,
                     "share": round(s.n_tracks / total, 4),
                     "material": s in material,
                     "date_min": s.date_min or s.date_min_all,
                     "date_max": s.date_max or s.date_max_all,
                     "band": s.describe()} for s in slist]
            for s in slist:
                sd: Dict[str, Any] = {
                    "n_tracks": s.n_tracks, "n_tracks_all_time": s.n_tracks_all_time,
                    "date_min": s.date_min, "date_max": s.date_max,
                    "date_min_all_time": s.date_min_all, "date_max_all_time": s.date_max_all,
                    "graded_min": s.graded_min, "graded_max": s.graded_max,
                    "centre_tol": s.centre_tol, "end_tol": s.end_tol,
                    "shape_ratio": s.shape_ratio, "step": s.step, "n_points": s.n_points,
                    "n_graded": s.n_graded, "is_flat": s.is_flat,
                    "graded_regions": s.graded_regions,
                    "raw_array_variants": s.raw_variants,
                    "findings": [],
                }
                if entry is not None and entry.tables:
                    t, how = pick_table(s, entry.tables)
                    sd["compared_to"] = {"role": t.role, "object": t.object_name,
                                         "source": t.source, "selected_by": how,
                                         "role_confidence": t.role_confidence,
                                         "band": t.describe()}
                    fs, info = compare(s, t, entry.tables)
                    if how != "role_match":
                        fs.insert(0, Finding(
                            "NO_TABLE_FOR_THIS_STATION", "medium",
                            f"the ATP has no table identifiable as the {station} table; "
                            f"compared against {_table_label(t)} instead, so every finding "
                            f"below is conditional on that being the right table"))
                    sd["axis"] = info
                    for f in fs:
                        sd["findings"].append(asdict(f))
                        rec["codes"].append(f.code)
                block["signatures"].append(sd)
            rec["stations"][station] = block
        rec["codes"] = sorted(set(rec["codes"]))
        results.append(rec)

    results.sort(key=lambda r: (-r["recent_tracks_total"], r["model"]))

    ext_counts = Counter(Path(s.path).suffix.lower() for s in sheets)
    status_counts = Counter(s.read_status for s in sheets)
    tables_by_ext = Counter()
    for s in sheets:
        if s.tables:
            tables_by_ext[Path(s.path).suffix.lower()] += len(s.tables)

    coverage = {
        "atp_files_scanned": len(sheets),
        "atp_files_by_ext": dict(ext_counts),
        "atp_files_by_read_status": dict(status_counts),
        "atp_tables_extracted": sum(len(s.tables) for s in sheets),
        "atp_tables_by_ext": dict(tables_by_ext),
        "atp_files_with_tables": sum(1 for s in sheets if s.tables),
        "atp_models_identified": len(index),
        "atp_models_with_tables": sum(1 for e in index.values() if e.tables),
        "skipped": dict(skipped),
        "db_models_with_limits": len(db_models),
        "db_models_matched_to_atp": len(matched),
        "db_models_unmatched": len(unmatched_db),
        "atp_models_unmatched": len(unmatched_atp),
        "model_specs_rows": len(specs),
        "since": since,
    }
    code_counts = Counter()
    for r in results:
        for c in r["codes"]:
            code_counts[c] += 1

    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "db": str(db_path), "atp_root": str(atp_root), "since": since,
        "coverage": coverage,
        "code_counts": dict(code_counts),
        "models": results,
        "unmatched_db_models": unmatched_db,
        "unmatched_atp_models": unmatched_atp,
        "near_misses": near_misses,
        "atp_sheets_unmatched": sorted(
            s.path for s in sheets
            if not any(mm in used_atp for mm in s.models)),
        "skipped_files": skipped_names,
    }


# =========================================================================== #
# 8. Output
# =========================================================================== #

CSV_MODEL_COLUMNS = [
    "model", "recent_tracks_total", "recent_tracks_trim", "recent_tracks_ft",
    "atp_model_key", "atp_revision", "atp_number", "customer",
    "atp_tables", "trim_signatures", "ft_signatures", "codes", "headline",
]

CSV_SIG_COLUMNS = [
    "model", "recent_tracks_total", "station", "n_tracks", "n_tracks_all_time",
    "date_min", "date_max", "station_graded_min", "station_graded_max",
    "station_centre_tol", "station_end_tol", "station_shape", "station_step",
    "station_points", "station_flat",
    "atp_role", "atp_role_confidence", "atp_graded_min", "atp_graded_max",
    "atp_centre_tol", "atp_end_tol", "atp_shape", "atp_step", "atp_points",
    "axis_kind", "value_ratio_min", "value_ratio_max", "codes", "detail",
]


def _headline(rec: Dict[str, Any]) -> str:
    order = ["TABLE_SWAP", "RANGE_MISMATCH", "SHAPE_MISMATCH", "VALUE_RATIO_VARIES",
             "MULTIPLE_TABLES_IN_USE", "VALUE_RATIO_CONSTANT_UNEXPLAINED",
             "INTERPOLATION_MISMATCH", "STATION_AXIS_NOT_MONOTONIC", "AXIS_UNRESOLVED",
             "MULTIPLE_TABLES_TAIL", "NO_TABLE_FOR_THIS_STATION", "NO_TABLE_IN_ATP", "NO_ATP"]
    for c in order:
        if c in rec["codes"]:
            return c
    if set(rec["codes"]) <= BENIGN_CODES:
        return "OK"
    return rec["codes"][0]


BENIGN_CODES = {"VALUE_RATIO_CLEAN", "SPEC_ROW_MISSING", "SPEC_RESISTANCE_MISSING",
                "SPEC_ANGLE_MISSING"}


def write_outputs(report: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "atp_spec_audit.csv",
        "signatures_csv": out_dir / "atp_spec_audit_signatures.csv",
        "json": out_dir / "atp_spec_audit.json",
        "md": out_dir / "atp_spec_audit.md",
    }
    with paths["csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_MODEL_COLUMNS)
        w.writeheader()
        for r in report["models"]:
            w.writerow({
                "model": r["model"],
                "recent_tracks_total": r["recent_tracks_total"],
                "recent_tracks_trim": r["recent_tracks_trim"],
                "recent_tracks_ft": r["recent_tracks_ft"],
                "atp_model_key": r["atp_model_key"] or "",
                "atp_revision": r["atp_revision"] or "",
                "atp_number": r["atp_number"] or "",
                "customer": r["customer"] or "",
                "atp_tables": len(r["atp_tables"]),
                "trim_signatures": len(r["stations"].get("trim", {}).get("signatures", [])),
                "ft_signatures": len(r["stations"].get("final_test", {}).get("signatures", [])),
                "codes": "|".join(r["codes"]),
                "headline": _headline(r),
            })
    with paths["signatures_csv"].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_SIG_COLUMNS)
        w.writeheader()
        for r in report["models"]:
            for station, block in r["stations"].items():
                for s in block.get("signatures", []):
                    ct = s.get("compared_to") or {}
                    atp = None
                    for t in r["atp_tables"]:
                        if t["object"] == ct.get("object") and t["source"] == ct.get("source"):
                            atp = t
                            break
                    ax = s.get("axis") or {}
                    w.writerow({
                        "model": r["model"], "recent_tracks_total": r["recent_tracks_total"],
                        "station": station, "n_tracks": s["n_tracks"],
                        "n_tracks_all_time": s["n_tracks_all_time"],
                        "date_min": s["date_min"] or s["date_min_all_time"] or "",
                        "date_max": s["date_max"] or s["date_max_all_time"] or "",
                        "station_graded_min": s["graded_min"], "station_graded_max": s["graded_max"],
                        "station_centre_tol": s["centre_tol"], "station_end_tol": s["end_tol"],
                        "station_shape": s["shape_ratio"], "station_step": s["step"],
                        "station_points": s["n_points"], "station_flat": s["is_flat"],
                        "atp_role": ct.get("role", ""),
                        "atp_role_confidence": ct.get("role_confidence", ""),
                        "atp_graded_min": atp["graded_min"] if atp else "",
                        "atp_graded_max": atp["graded_max"] if atp else "",
                        "atp_centre_tol": atp["centre_tol"] if atp else "",
                        "atp_end_tol": atp["end_tol"] if atp else "",
                        "atp_shape": atp["shape_ratio"] if atp else "",
                        "atp_step": atp["step"] if atp else "",
                        "atp_points": atp["n_points"] if atp else "",
                        "axis_kind": ax.get("axis_kind", ""),
                        "value_ratio_min": ax.get("value_ratio_min", ""),
                        "value_ratio_max": ax.get("value_ratio_max", ""),
                        "codes": "|".join(sorted({f["code"] for f in s["findings"]})),
                        "detail": " | ".join(f["detail"] for f in s["findings"])[:900],
                    })
    paths["json"].write_text(json.dumps(report, indent=1, default=str))
    paths["md"].write_text(render_markdown(report))
    return paths


def _g(v: Any) -> str:
    """Compact number for the report -- 0.0029999999999999996 reads as 0.003."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_markdown(report: Dict[str, Any]) -> str:
    cov = report["coverage"]
    L: List[str] = []
    L.append("# ATP vs station linearity spec audit")
    L.append("")
    L.append(f"Generated {report['generated']} · DB `{report['db']}` · ATP root "
             f"`{report['atp_root']}` · recent window from **{report['since']}**")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("*Centre* tolerance is the band at the graded point nearest the middle of the "
             "graded window; *end* is the larger of the two graded extremes; *shape* is "
             "end/centre. Shape is the only comparison that needs no axis alignment, so it is "
             "the one to trust when the two position columns disagree. For a stepped band "
             "(0.002 / 0.005 / 0.001-at-the-index-point) centre/end describe only two of the "
             "levels -- read `VALUE_RATIO_*`, which compares every matched angle.")
    L.append("")
    L.append("| code | means |")
    L.append("|---|---|")
    for code, meaning in (
        ("RANGE_MISMATCH", "one side grades angles the other marks not-graded. "
                           "Unit-independent; high confidence unless the axis had to be rescaled"),
        ("SHAPE_MISMATCH", "end/centre ratios differ by more than 5%. Unit-independent"),
        ("TABLE_SWAP", "the station's shape matches a *different* table in the same ATP"),
        ("VALUE_RATIO_CLEAN", "station = ATP x a constant, and that constant is one of "
                              "1, 2, 5, 10, 0.5, 0.2, 0.1 (a V.R.-vs-volts unit difference)"),
        ("VALUE_RATIO_CONSTANT_UNEXPLAINED", "a constant scale, but not one of those"),
        ("VALUE_RATIO_VARIES", "the ratio changes across the travel: different curves, "
                               "not the same curve in different units"),
        ("INTERPOLATION_MISMATCH", "station points between ATP grid angles are not the "
                                   "linear interpolation of their neighbours"),
        ("MULTIPLE_TABLES_IN_USE", "two or more limit bands in production for one model and "
                                   "station, each with >=10 tracks and >=5% of the volume"),
        ("MULTIPLE_TABLES_TAIL", "extra bands exist but only as a small tail"),
        ("STATION_AXIS_NOT_MONOTONIC", "the ingested position column does not sweep -- it is "
                                       "not a travel axis, and nothing angle-based can be said"),
        ("AXIS_UNRESOLVED", "ATP and station sweep different quantities (spans differ by "
                            ">50%); only the shape check was run"),
        ("AXIS_RESCALED", "different units; angles compared proportionally and approximate"),
        ("NO_TABLE_FOR_THIS_STATION", "the ATP has tables but none identifiable as this "
                                      "station's; compared against another one"),
        ("NO_TABLE_IN_ATP", "an ATP was matched but it carries no per-point linearity table"),
        ("NO_ATP", "no ATP sheet in the library matched this model"),
        ("SPEC_ROW_MISSING", "the model has no row in the `model_specs` table"),
    ):
        L.append(f"| `{code}` | {meaning} |")
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append("| what | count |")
    L.append("|---|---:|")
    L.append(f"| ATP files scanned | {cov['atp_files_scanned']} |")
    for ext, n in sorted(cov["atp_files_by_ext"].items()):
        tabs = cov["atp_tables_by_ext"].get(ext, 0)
        L.append(f"| &nbsp;&nbsp;`{ext}` files | {n} (tables extracted: {tabs}) |")
    L.append(f"| ATP files with >=1 linearity table | {cov['atp_files_with_tables']} |")
    L.append(f"| linearity tables extracted | {cov['atp_tables_extracted']} |")
    L.append(f"| models identified in the ATP library | {cov['atp_models_identified']} |")
    L.append(f"| &nbsp;&nbsp;of those, with a table | {cov['atp_models_with_tables']} |")
    L.append(f"| DB models with station limit arrays | {cov['db_models_with_limits']} |")
    L.append(f"| DB models matched to an ATP | {cov['db_models_matched_to_atp']} |")
    L.append(f"| DB models with NO ATP | {cov['db_models_unmatched']} |")
    L.append(f"| ATP models never seen in the DB | {cov['atp_models_unmatched']} |")
    L.append(f"| model_specs rows | {cov['model_specs_rows']} |")
    for k, v in sorted(cov["skipped"].items()):
        L.append(f"| skipped: {k} | {v} |")
    L.append("")
    L.append("## Findings by code")
    L.append("")
    L.append("| code | models |")
    L.append("|---|---:|")
    for code, n in sorted(report["code_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| `{code}` | {n} |")
    L.append("")
    L.append("## Models by production volume (recent window)")
    L.append("")
    L.append("| # | model | recent tracks (trim/FT) | ATP | headline | codes |")
    L.append("|---:|---|---:|---|---|---|")
    for i, r in enumerate(report["models"][:80], 1):
        L.append(f"| {i} | `{r['model']}` | {r['recent_tracks_total']} "
                 f"({r['recent_tracks_trim']}/{r['recent_tracks_ft']}) | "
                 f"{r['atp_model_key'] or '-'} rev {r['atp_revision'] or '?'} | "
                 f"`{_headline(r)}` | {', '.join(r['codes']) or '-'} |")
    L.append("")
    L.append("## Detail")
    L.append("")
    for r in report["models"]:
        if not r["codes"] or set(r["codes"]) <= BENIGN_CODES:
            continue
        L.append(f"### {r['model']} — {r['recent_tracks_total']} recent tracks "
                 f"({r['recent_tracks_trim']} trim / {r['recent_tracks_ft']} FT)")
        L.append("")
        if r["atp_model_key"]:
            L.append(f"ATP: `{r['atp_model_key']}` rev {r['atp_revision'] or '?'} "
                     f"{r['atp_number'] or ''} · customer {r['customer'] or '?'}")
            for s in r["atp_sheets"][:4]:
                L.append(f"  - sheet: `{s}`")
            for t in r["atp_tables"]:
                head = (f"  - table [{t['role']}/{t['role_confidence']}] "
                        f"x=`{t['angle_header']}` value=`{t['value_header']}` "
                        f"{_g(t['value_min'])}..{_g(t['value_max'])}")
                if t.get("shape_ratio"):
                    head += (f" — graded {_g(t['graded_min'])}..{_g(t['graded_max'])}, "
                             f"centre {_g(t['centre_tol'])}, end {_g(t['end_tol'])}, "
                             f"shape {t['shape_ratio']:.4g}, {t['n_graded']}/{t['n_points']} pts, "
                             f"step {_g(t['step'])}"
                             + (", FLAT" if t["is_flat"] else ""))
                else:
                    head += " — no graded points"
                L.append(head)
                L.append(f"    evidence: {t['role_evidence']}")
        else:
            L.append("ATP: **none matched**")
        L.append("")
        for station, block in r["stations"].items():
            if block.get("status") == "NO_STATION_DATA":
                continue
            L.append(f"**{station}**" + ("  _(nothing in the recent window; showing the "
                                         "last bands this model ran on)_"
                                         if block.get("status") == "stale" else ""))
            for s in block["signatures"]:
                line = (f"  - {s['n_tracks']} recent tracks ({s['n_tracks_all_time']} all-time), "
                        f"{(s['date_min'] or s['date_min_all_time'] or '?')[:10]}.."
                        f"{(s['date_max'] or s['date_max_all_time'] or '?')[:10]}: graded "
                        f"{_g(s['graded_min'])}..{_g(s['graded_max'])}, "
                        f"centre {_g(s['centre_tol'])}, end {_g(s['end_tol'])}")
                if s.get("shape_ratio"):
                    line += f", shape {s['shape_ratio']:.4g}"
                line += (f", {s['n_graded']}/{s['n_points']} pts, step {_g(s['step'])}"
                         + (", FLAT" if s["is_flat"] else ""))
                L.append(line)
                live = [v for v in s.get("raw_array_variants", []) if v["n_tracks"] > 0]
                if len(live) > 1:
                    L.append("    raw array variants in window: " + "; ".join(
                        f"{v['n_tracks']}x {v['n_points']}pts" for v in live))
                for f in s["findings"]:
                    L.append(f"    - **{f['code']}** ({f['severity']}): {f['detail']}")
            L.append("")
        for f in r["prose_findings"]:
            L.append(f"  - **{f['code']}**: {f['detail']}")
        L.append("")
    L.append("## Unmatched")
    L.append("")
    L.append(f"### DB models with no ATP ({len(report['unmatched_db_models'])})")
    L.append("")
    L.append(", ".join(f"`{m}`" for m in report["unmatched_db_models"]) or "_none_")
    L.append("")
    L.append(f"### ATP models never seen in the DB ({len(report['unmatched_atp_models'])})")
    L.append("")
    L.append(", ".join(f"`{m}`" for m in report["unmatched_atp_models"]) or "_none_")
    L.append("")
    L.append(f"### Near misses for a human to resolve ({len(report['near_misses'])})")
    L.append("")
    L.append("These were **not** matched. A wrong match is worse than an unmatched one.")
    L.append("")
    for nm in report["near_misses"]:
        L.append(f"  - DB `{nm['db_model']}` (normalised `{nm['normalised']}`) ~ "
                 + ", ".join(f"`{c}`" for c in nm["atp_candidates"][:6]))
    L.append("")
    return "\n".join(L)


# =========================================================================== #
# 9. Self-test -- the 8232-1 reference case
# =========================================================================== #

REF_MODEL = "8232-1"
REF_SINCE = "2024-01-01"
REF_DOC = "8200's/8232-1 Datasheet Rev M.doc"


def _approx(a: Optional[float], b: float, tol: float) -> bool:
    return a is not None and abs(a - b) <= tol


def _count_with_clean_positions(db_path: Path, model: str, since: str,
                                limits: Sequence[Optional[float]]) -> int:
    """Tracks on one FT limit array whose angle column carries no float noise.

    Only used by the self-test, to keep the quoted 1,935 figure reproducible
    alongside the 2,254 the band grouping reports.
    """
    want = tuple(None if v is None else round(v, 6) for v in limits)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    n = 0
    try:
        cur = con.cursor()
        cur.execute("""SELECT t.position_data, t.upper_limits FROM final_test_tracks t
                       JOIN final_test_results f ON f.id = t.final_test_id
                       WHERE f.model = ? AND f.file_date >= ?""", (model, since))
        for pos_s, lim_s in cur.fetchall():
            lims = _json_floats(lim_s, absolute=True)
            poss = _json_floats(pos_s, absolute=False)
            if not lims or not poss:
                continue
            if tuple(None if v is None else round(v, 6) for v in lims) != want:
                continue
            if all(p is not None and abs(p - round(p)) < 1e-9 for p in poss[1:-1]):
                n += 1
    finally:
        con.close()
    return n


def self_test(db_path: Path, atp_root: Path) -> int:
    """Reproduce the verified 8232-1 reference numbers.  Non-zero exit on any diff."""
    fails: List[str] = []
    checks = 0

    def check(name: str, ok: bool, got: Any, want: Any) -> None:
        nonlocal checks
        checks += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got {got!r}, want {want!r}")
        if not ok:
            fails.append(name)

    print(f"self-test: ATP reference case {REF_MODEL} ({REF_DOC})")
    doc = atp_root / REF_DOC
    if not doc.exists():
        print(f"  [FAIL] reference ATP not found at {doc}")
        return 1
    sheet = parse_sheet(doc, atp_root)

    check("ATP model line normalises to 8232-1", sheet.text_models[:1] == ["8232-1"],
          sheet.text_models[:1], ["8232-1"])
    check("ATP revision is M", sheet.revision == "M", sheet.revision, "M")
    check("ATP number is ATP-10396-DS", (sheet.atp_number or "").startswith("ATP-10396"),
          sheet.atp_number, "ATP-10396-DS")
    check("customer is Goodrich", (sheet.customer or "").lower().startswith("goodrich"),
          sheet.customer, "Goodrich")
    check("resistance 5250 +/- 250 ohms",
          _approx(sheet.resistance_nominal, 5250, 1) and _approx(sheet.resistance_tol, 250, 1),
          (sheet.resistance_nominal, sheet.resistance_tol), (5250.0, 250.0))
    check("electrical angle 27.5", _approx(sheet.electrical_angle, 27.5, 0.01),
          sheet.electrical_angle, 27.5)
    check("linearity type Absolute", sheet.linearity_type == "Absolute",
          sheet.linearity_type, "Absolute")
    check("output smoothness 0.05% max", (sheet.smoothness or "").startswith("0.05"),
          sheet.smoothness, "0.05% max.")
    check("two linearity tables extracted", len(sheet.tables) == 2, len(sheet.tables), 2)

    trim_t = next((t for t in sheet.tables if "v.r" in t.value_header), None)
    ft_t = next((t for t in sheet.tables if "volt" in t.value_header), None)
    check("found the Theo. V.R. (trim) table", trim_t is not None,
          trim_t.value_header if trim_t else None, "theo. v.r.")
    check("found the Voltage (final-test) table", ft_t is not None,
          ft_t.value_header if ft_t else None, "voltage")

    if trim_t:
        check("trim table value column 0..1",
              _approx(trim_t.value_min, 0.0, 1e-9) and _approx(trim_t.value_max, 1.0, 1e-9),
              (trim_t.value_min, trim_t.value_max), (0.0, 1.0))
        check("trim table graded -22..22",
              _approx(trim_t.graded_min, -22, 1e-9) and _approx(trim_t.graded_max, 22, 1e-9),
              (trim_t.graded_min, trim_t.graded_max), (-22, 22))
        check("trim table centre tolerance 0.0030", _approx(trim_t.centre_tol, 0.0030, 1e-9),
              trim_t.centre_tol, 0.0030)
        check("trim table end tolerance 0.0099993", _approx(trim_t.end_tol, 0.0099993, 1e-9),
              trim_t.end_tol, 0.0099993)
        check("trim table shape 3.33x", _approx(trim_t.shape_ratio, 3.3331, 5e-4),
              round(trim_t.shape_ratio, 4) if trim_t.shape_ratio else None, 3.3331)
        check("trim table step 2 degrees", _approx(trim_t.step, 2.0, 1e-9), trim_t.step, 2.0)
        check("trim table 29 graded points", trim_t.n_graded == 23, trim_t.n_graded, 23)
        check("trim table 29 printed points", trim_t.n_points == 29, trim_t.n_points, 29)
        check("trim table role guess = trim", trim_t.role == "trim", trim_t.role, "trim")
        check("trim table role confidence high", trim_t.role_confidence == "high",
              trim_t.role_confidence, "high")
    if ft_t:
        check("FT table value column 0..5",
              _approx(ft_t.value_min, 0.0, 1e-9) and _approx(ft_t.value_max, 5.0, 1e-9),
              (ft_t.value_min, ft_t.value_max), (0.0, 5.0))
        check("FT table graded -22..22",
              _approx(ft_t.graded_min, -22, 1e-9) and _approx(ft_t.graded_max, 22, 1e-9),
              (ft_t.graded_min, ft_t.graded_max), (-22, 22))
        check("FT table centre tolerance 0.0030", _approx(ft_t.centre_tol, 0.0030, 1e-9),
              ft_t.centre_tol, 0.0030)
        check("FT table end tolerance 0.0064996", _approx(ft_t.end_tol, 0.00649965, 1e-8),
              ft_t.end_tol, 0.00649965)
        check("FT table shape 2.17x", _approx(ft_t.shape_ratio, 2.1666, 5e-4),
              round(ft_t.shape_ratio, 4) if ft_t.shape_ratio else None, 2.1666)
        check("FT table role guess = final_test", ft_t.role == "final_test", ft_t.role,
              "final_test")

    print(f"self-test: station side, {REF_MODEL} from {REF_SINCE}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sigs = collect_station_signatures(con, REF_SINCE, models=[REF_MODEL])
    finally:
        con.close()
    # bands actually in use inside the reference window (historical bands from
    # 2010-2019 are still collected, and still reported, but are not "in use")
    trim_sigs = [s for s in sigs.get((REF_MODEL, "trim"), []) if s.n_tracks > 0]
    ft_sigs = [s for s in sigs.get((REF_MODEL, "final_test"), []) if s.n_tracks > 0]
    check("one trim band in use", len(trim_sigs) == 1, len(trim_sigs), 1)
    if trim_sigs:
        s = trim_sigs[0]
        check("trim band on 1,684 tracks since 2024-01", s.n_tracks == 1684, s.n_tracks, 1684)
        check("trim band graded -22..22",
              _approx(s.graded_min, -22, 1e-9) and _approx(s.graded_max, 22, 1e-9),
              (s.graded_min, s.graded_max), (-22, 22))
        check("trim band centre 0.0055", _approx(s.centre_tol, 0.0055, 1e-9), s.centre_tol, 0.0055)
        check("trim band end 0.0169074", _approx(s.end_tol, 0.0169074, 1e-6), s.end_tol, 0.0169074)
        check("trim band shape 3.07x", _approx(s.shape_ratio, 3.0741, 1e-3),
              round(s.shape_ratio, 4), 3.0741)
        grids = sorted(v["n_points"] for v in s.raw_variants if v["n_tracks"] > 0)
        check("trim band has 2 raw-array grids in the window (57 and 111 pts)",
              grids == [57, 111], grids, [57, 111])
    check("3 distinct FT bands in use", len(ft_sigs) == 3, len(ft_sigs), 3)
    if ft_sigs:
        a = ft_sigs[0]
        check("FT dominant band centre 0.0030", _approx(a.centre_tol, 0.0030, 1e-9),
              a.centre_tol, 0.0030)
        check("FT dominant band end 0.0100", _approx(a.end_tol, 0.0100, 1e-9), a.end_tol, 0.0100)
        check("FT dominant band shape 3.33x", _approx(a.shape_ratio, 3.3333, 1e-3),
              round(a.shape_ratio, 4), 3.3333)
        check("FT dominant band graded -28..28",
              _approx(a.graded_min, -28, 0.05) and _approx(a.graded_max, 28, 0.05),
              (a.graded_min, a.graded_max), (-28, 28))
        # Reconciliation of the 2,254 reported here against the 1,935 quoted
        # earlier for this same band (see the module docstring of the finding):
        #   2,254 = every track graded to the 0.0030 -> 0.0100 bowtie
        #   1,934 = those whose angle column is noise-free (319 tracks write
        #           -27.9835 where the next unit writes -28.0, with a
        #           byte-identical limit array, so a position-inclusive
        #           signature splits them off)
        #     +28 = the 0.0025 -> 0.0095 band, a different spec
        #      +1 = one 55-point track graded -28..+26 (centre 0.003318), a
        #           different graded window again
        #   1,963 = 1,934 + 28 + 1, all the noise-free tracks
        # 1,935 is 1,934 + that single truncated track; there is no definition
        # under which the 0.0030 band alone holds 1,935 tracks.
        check("FT dominant band on 2,254 tracks since 2024-01", a.n_tracks == 2254,
              a.n_tracks, 2254)
        check("FT dominant band is one raw limit array (57 pts)",
              [v["n_points"] for v in a.raw_variants if v["n_tracks"] > 0] == [57],
              [v["n_points"] for v in a.raw_variants if v["n_tracks"] > 0], [57])
        legacy = _count_with_clean_positions(db_path, REF_MODEL, REF_SINCE, a.limits)
        check("...of which 1,934 have noise-free angles (1,935 was 1,934 + one "
              "truncated -28..+26 track)", legacy == 1934, legacy, 1934)
        if len(ft_sigs) > 1:
            b = ft_sigs[1]
            check("FT second band on 28 tracks", b.n_tracks == 28, b.n_tracks, 28)
            check("FT second band centre 0.0025", _approx(b.centre_tol, 0.0025, 1e-9),
                  b.centre_tol, 0.0025)
            check("FT second band end 0.0095", _approx(b.end_tol, 0.0095, 1e-9), b.end_tol, 0.0095)
            check("FT second band shape 3.80x", _approx(b.shape_ratio, 3.8, 1e-3),
                  round(b.shape_ratio, 4), 3.8)
            check("FT second band first seen 2025-08", (b.date_min or "").startswith("2025-08"),
                  b.date_min, "2025-08-22")

    print("self-test: verdicts")
    if trim_sigs and ft_sigs and trim_t and ft_t:
        tf, _ = compare(trim_sigs[0], trim_t, sheet.tables)
        codes = {f.code for f in tf}
        check("trim vs ATP trim table: no RANGE_MISMATCH (both grade +/-22)",
              "RANGE_MISMATCH" not in codes, sorted(codes), "no RANGE_MISMATCH")
        check("trim vs ATP trim table: SHAPE_MISMATCH (3.07 vs 3.33)",
              "SHAPE_MISMATCH" in codes, sorted(codes), "SHAPE_MISMATCH")
        check("trim vs ATP trim table: VALUE_RATIO_VARIES (1.83 centre, 1.69 end)",
              "VALUE_RATIO_VARIES" in codes, sorted(codes), "VALUE_RATIO_VARIES")
        picked, how = pick_table(ft_sigs[0], sheet.tables)
        check("FT compares against the ATP's final-test table",
              picked is ft_t and how == "role_match",
              (picked.value_header if picked else None, how), ("voltage", "role_match"))
        ff, _ = compare(ft_sigs[0], ft_t, sheet.tables)
        fcodes = {f.code for f in ff}
        check("FT vs ATP FT table: TABLE_SWAP (station shape matches the ATP TRIM table)",
              "TABLE_SWAP" in fcodes, sorted(fcodes), "TABLE_SWAP")
        check("FT vs ATP FT table: RANGE_MISMATCH (station grades +/-23..28, ATP does not)",
              "RANGE_MISMATCH" in fcodes, sorted(fcodes), "RANGE_MISMATCH")
        swap = next((f for f in ff if f.code == "TABLE_SWAP"), None)
        if swap:
            check("TABLE_SWAP names the trim table",
                  swap.data.get("matched_table_role") == "trim",
                  swap.data.get("matched_table_role"), "trim")
        rng = next((f for f in ff if f.code == "RANGE_MISMATCH"), None)
        if rng:
            check("RANGE_MISMATCH counts 12 station-only points per unit",
                  rng.data.get("station_only_points") == 12,
                  rng.data.get("station_only_points"), 12)
        # and, against the ATP TRIM table, the FT station is a 1:1 copy
        ff2, _ = compare(ft_sigs[0], trim_t, sheet.tables)
        f2codes = {f.code for f in ff2}
        check("FT vs ATP TRIM table: clean x1 value ratio",
              "VALUE_RATIO_CLEAN" in f2codes, sorted(f2codes), "VALUE_RATIO_CLEAN")
        check("FT vs ATP TRIM table: no SHAPE_MISMATCH",
              "SHAPE_MISMATCH" not in f2codes, sorted(f2codes), "no SHAPE_MISMATCH")

    print()
    print(f"self-test: {checks - len(fails)}/{checks} checks passed")
    if fails:
        print("FAILED: " + ", ".join(fails))
        return len(fails)
    return 0


# =========================================================================== #
# CLI
# =========================================================================== #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="atp_spec_audit.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("db", type=Path,
                    help="path to the analysis database (opened READ-ONLY; "
                         "data/analysis.db is fine)")
    ap.add_argument("--atp-root", type=Path, default=Path("data/Work files/DATA SHEETS"),
                    help="root of the ATP document library (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=Path("qa_output/atp_audit"),
                    help="where the CSV/JSON/Markdown go (default: %(default)s)")
    ap.add_argument("--months", type=int, default=24,
                    help="size of the 'recent' production window, in months (default: %(default)s)")
    ap.add_argument("--since", type=str, default=None,
                    help="explicit YYYY-MM-DD start of the recent window (overrides --months)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only read the first N ATP files (debugging)")
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce the verified 8232-1 reference numbers and exit non-zero "
                         "if any differ")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 2
    if not args.atp_root.exists():
        print(f"ATP root not found: {args.atp_root}", file=sys.stderr)
        return 2
    if _require_olefile() is None:
        return 2

    if args.self_test:
        return self_test(args.db, args.atp_root)

    since = args.since or months_ago(args.months)
    report = run_audit(args.db, args.atp_root, since, limit=args.limit)
    paths = write_outputs(report, args.out_dir)

    cov = report["coverage"]
    print()
    print("ATP files scanned          :", cov["atp_files_scanned"])
    print("linearity tables extracted :", cov["atp_tables_extracted"],
          f"(from {cov['atp_files_with_tables']} files)")
    print("ATP models identified      :", cov["atp_models_identified"],
          f"({cov['atp_models_with_tables']} with a table)")
    print("DB models with limits      :", cov["db_models_with_limits"])
    print("  matched to an ATP        :", cov["db_models_matched_to_atp"])
    print("  no ATP                   :", cov["db_models_unmatched"])
    print("ATP models unused by DB    :", cov["atp_models_unmatched"])
    print()
    for code, n in sorted(report["code_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {code:36s} {n} models")
    print()
    for k, p in paths.items():
        print(f"  {k:16s} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
