"""Check our per-event targets against the organisers' label file. Re-run after
any change to ingest or features.

    .venv/bin/python scripts/check_labels.py [path]

Reads the organisers' label file from data/labels/ or data/ (.csv, or the
.xlsx, which needs openpyxl) and asserts, event by event, that build_event_table gives the same
training set (usable_for_training), final risk, final miss distance and final
tier. Exits non-zero on any mismatch.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import build_event_table  # noqa: E402
from src.ingest import DATA_DIR, load_split  # noqa: E402

# Wherever a teammate put the organisers' file; a .csv export avoids openpyxl
CANDIDATES = [DATA_DIR / d / f"{n}.{x}" for x in ("csv", "xlsx") for d in ("labels", ".")
              for n in ("challenge-1-event-labels", "challenge1_event_labels")]


def find_labels() -> Path | None:
    return next((p for p in CANDIDATES if p.exists()), None)


def load_labels(path: Path | None = None) -> pd.DataFrame:
    path = path or find_labels()
    if path is None:
        sys.exit("no label file: put challenge-1-event-labels.xlsx in data/labels/")
    if path.suffix == ".xlsx":
        # Row 1 is a description, row 2 blank, row 3 the header
        lab = pd.read_excel(path, sheet_name="train_event_labels", header=2)
    else:
        lab = pd.read_csv(path)
    lab["usable"] = lab["usable_for_training"].eq("yes")
    lab["final_tier"] = lab["final_tier"].str.upper()
    return lab.set_index("event_id")


def compare(ev: pd.DataFrame, lab: pd.DataFrame) -> list[str]:
    ev = ev[ev["split"] == "train"].set_index("event_id")
    errors = []
    # Events with no CDM at >= 2 d have no feature row; the file marks them usable=no
    missing = lab.index.difference(ev.index)
    if lab.loc[missing, "usable"].any():
        bad = missing[lab.loc[missing, "usable"].to_numpy()]
        errors.append(f"{len(bad)} usable events have no feature row (e.g. {list(bad[:5])})")
    extra = ev.index.difference(lab.index)
    if len(extra):
        errors.append(f"{len(extra)} events not in the label file (e.g. {list(extra[:5])})")

    both = ev.index.intersection(lab.index)
    e, l = ev.loc[both], lab.loc[both]
    checks = {
        "has_target != usable_for_training": e["has_target"] != l["usable"],
        "risk_final differs (> 1e-3)": l["usable"] & ~np.isclose(
            e["risk_final"], l["final_risk_log10_pc"], atol=1e-3),
        "miss_final_m differs (> 1 m)": l["usable"] & ~np.isclose(
            e["miss_final_m"], l["final_miss_distance_m"], atol=1.0),
        "tier_actual differs": l["usable"] & (e["tier_actual"] != l["final_tier"]),
    }
    for name, bad in checks.items():
        if bad.any():
            errors.append(f"{int(bad.sum())} events: {name} (e.g. {list(bad[bad].index[:5])})")
    return errors


def main():
    lab = load_labels(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    ev = build_event_table(load_split("train"))
    used = lab[lab["usable"]]
    print(f"label file: {len(lab):,} events, {len(used):,} usable; "
          f"usable tiers {used['final_tier'].value_counts().to_dict()}")
    print(f"ours: {int(ev['has_target'].sum()):,} events with a target; "
          f"tiers {ev.loc[ev['has_target'], 'tier_actual'].value_counts().to_dict()}")
    errors = compare(ev, lab)
    for msg in errors:
        print("  FAIL", msg)
    if errors:
        sys.exit(1)
    print("  ok   targets match the organisers' labels event by event")


if __name__ == "__main__":
    main()
