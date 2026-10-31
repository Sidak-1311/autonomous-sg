"""Load and validate conjunction data messages (CDMs).

One row per CDM, many CDMs per event. Broken rows are logged and dropped,
never silently fixed; gaps that are expected (sigmas, space weather) are
filled and flagged.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.tiering import RISK_FLOOR, tier_array

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

COLUMNS = [
    "event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance",
    "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "c_object_type", "c_sigma_r", "t_sigma_r", "F10", "AP",
]
# Object size, area-to-mass and orbit, used only by the debris-cascade model (src/cascade.py).
# Left as-is (NaN allowed): cascade() returns NaN where they are missing.
CASCADE_COLUMNS = [
    "mission_id", "t_span", "c_span", "t_cd_area_over_mass", "c_cd_area_over_mass", "c_rcs_estimate",
    "t_h_per", "t_h_apo", "c_h_per", "c_h_apo",
]
COLUMNS = COLUMNS + CASCADE_COLUMNS
OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN"]


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{split}_data.csv", usecols=COLUMNS)
    df = validate(df, split)
    df.insert(0, "split", split)
    return df.sort_values(["event_id", "time_to_tca"], ascending=[True, False],
                          ignore_index=True)


def validate(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    df = df.copy()
    bad = (
        df[["event_id", "time_to_tca", "risk", "miss_distance"]].isna().any(axis=1)
        | (df["miss_distance"] < 0)
        | (df["c_sigma_r"] < 0)
        | (df["t_sigma_r"] < 0)
        | ~df["risk"].between(RISK_FLOOR, 0)
    )
    if bad.any():
        log.warning("%s: dropping %d invalid rows", name, int(bad.sum()))
        df = df[~bad]

    # "TBA" is an object not yet catalogued, same as unknown for tracking quality
    df["c_object_type"] = (df["c_object_type"].fillna("UNKNOWN")
                           .replace({"TBA": "UNKNOWN"}))
    df.loc[~df["c_object_type"].isin(OBJECT_TYPES), "c_object_type"] = "UNKNOWN"

    for col in ["c_sigma_r", "t_sigma_r"]:
        missing = df[col].isna()
        df[f"{col}_missing"] = missing.astype(np.int8)
        df[col] = df[col].fillna(df.groupby("c_object_type")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # Space weather is shared by all CDMs of an event; fill within the event first
    df["sw_missing"] = df["F10"].isna().astype(np.int8)
    for col in ["F10", "AP"]:
        df[col] = df.groupby("event_id")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())

    df["tier"] = tier_array(df["risk"], df["miss_distance"])
    return df


def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.concat([load_split("train", data_dir), load_split("test", data_dir)],
                     ignore_index=True)
