"""Dataset discovery: find EDFs, parse subject + experimental/control group.

Naming scheme (as used in this project):
    <subject><E|C><session-id>[_p<part>].edf
    e.g.  nibaE00.edf  -> subject "niba", group "E" (experimental / person who stutters)
          vimaC02.edf  -> subject "vima", group "C" (control)
          vimaC02_p2.edf -> subject "vima", group "C", recording part 2 (crash-recovered)

The data directory is NOT hardcoded: it is resolved from (1) an explicit argument,
(2) the EEG_DATA_DIR environment variable, or (3) the ``data/`` folder next to the
repo root.  Point any of these at your own recordings that follow the scheme above.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# scheme: letters (subject), one E/C (group), digits (session id), optional _pN (part)
_NAME = re.compile(
    r"^(?P<subject>[A-Za-z]+?)(?P<group>[EC])(?P<sid>\d+)"
    r"(?:_p(?P<part>\d+))?\.edf$", re.IGNORECASE)

GROUP_LABEL = {"E": "experimental", "C": "control"}

# --- transparent, committed dataset annotations (NOT hidden scripts) --------------
# Some files (crash-recovered) have no stage markers.  Their boundaries were estimated
# from the data (eyes-closed alpha window + hyperventilation breathing surge) and should
# be verified against the video/log.  This is dataset metadata; a visitor with marker'd
# files never needs it, and it is trivial to delete or extend.
STAGE_OVERRIDES = {
    "vimaC02": [  # recovered part 1, recorded 2026-09-17; NO markers in file
        ("rest_open1", 60.0, 210.0),
        ("rest_close1", 210.0, 400.0),
        ("hypervent", 400.0, 580.0),
        ("rest_open2", 580.0, 760.0),
        ("reading", 760.0, 940.0),
        ("rest_open3", 940.0, 1120.0),
        ("speech", 1120.0, 1385.0),   # recording crashed shortly after speech start
    ],
}


def repo_data_dir() -> Path:
    """Default: <repo_root>/data  (repo_root = parent of this package)."""
    return Path(__file__).resolve().parent.parent / "data"


def resolve_dir(data_dir: str | os.PathLike | None = None) -> Path:
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    env = os.environ.get("EEG_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_data_dir()


def parse_name(basename: str) -> dict | None:
    m = _NAME.match(basename)
    if not m:
        return None
    d = m.groupdict()
    d["group"] = d["group"].upper()
    d["part"] = int(d["part"]) if d.get("part") else None
    d["group_label"] = GROUP_LABEL.get(d["group"], "?")
    return d


def discover(data_dir: str | os.PathLike | None = None) -> list[dict]:
    """Return records (dicts) for every EDF matching the naming scheme, sorted.

    Each record: path, basename, stem, subject, group, group_label, sid, part, key.
    ``key`` = basename without .edf (unique per recording; used for overrides).
    """
    d = resolve_dir(data_dir)
    recs = []
    for p in sorted(d.glob("*.edf")):
        info = parse_name(p.name)
        if info is None:
            continue
        rec = dict(info)
        rec["path"] = str(p)
        rec["basename"] = p.name
        rec["stem"] = p.stem
        rec["key"] = p.stem
        recs.append(rec)
    recs.sort(key=lambda r: (r["subject"], r["sid"] or "", r["part"] or 0))
    return recs


def summarize(recs: list[dict]) -> str:
    lines = [f"Discovered {len(recs)} recording(s):"]
    for r in recs:
        part = f" [part {r['part']}]" if r["part"] else ""
        lines.append(f"  {r['basename']:<16} subject={r['subject']:<6} "
                     f"group={r['group']} ({r['group_label']}){part}")
    subj = {}
    for r in recs:
        subj.setdefault((r["subject"], r["group"]), []).append(r["stem"])
    lines.append("\nSubjects (same person = same subject prefix):")
    for (s, g), stems in subj.items():
        tag = GROUP_LABEL[g]
        lines.append(f"  {s} [{g}={tag}]: {', '.join(stems)}")
    return "\n".join(lines)
