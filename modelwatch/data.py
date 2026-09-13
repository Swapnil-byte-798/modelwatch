"""ACS corpus construction.

Downloads ACS PUMS person files one (state, year) cell at a time, extracts the
ACSIncome feature set, subsamples to a fixed n, writes parquet, and then deletes
the raw download. The delete matters: the full 2014-2018 PUMS pull is tens of
gigabytes, and we only ever need WINDOW_N rows per cell.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome

from . import config as C

# 50 states + DC + PR
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR",
]


def window_path(state: str, year: int) -> Path:
    return C.WINDOWS_DIR / f"{state}_{year}.parquet"


def _purge_cache() -> None:
    """Delete raw census downloads. Called after every cell."""
    if C.CACHE_DIR.exists():
        shutil.rmtree(C.CACHE_DIR, ignore_errors=True)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_cell(state: str, year: int, n: int | None = None, force: bool = False) -> dict:
    """Materialise one (state, year) window as parquet. Returns a status dict."""
    n = n or C.WINDOW_N
    out = window_path(state, year)
    if out.exists() and not force:
        rows = pd.read_parquet(out).shape[0]
        return {"state": state, "year": year, "status": "cached", "rows": rows}

    try:
        source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person",
            root_dir=str(C.CACHE_DIR),
        )
        raw = source.get_data(states=[state], download=True)
        feats, labels, _ = ACSIncome.df_to_pandas(raw)
    except Exception as exc:  # noqa: BLE001 - we want the cell name in the message
        _purge_cache()
        return {"state": state, "year": year, "status": "error", "rows": 0,
                "detail": f"{type(exc).__name__}: {exc}"}

    df = feats[C.FEATURES].copy()
    df["y"] = labels.to_numpy().astype(int).ravel()
    available = len(df)

    if available < n:
        _purge_cache()
        return {"state": state, "year": year, "status": "too_small",
                "rows": available}

    # Fixed-n subsample. Seeded per cell so the corpus is reproducible but each
    # cell draws independently. Uses a stable digest rather than hash(): Python
    # randomises string hashing per process, which would make the corpus
    # irreproducible across runs -- the one thing this project cannot afford.
    digest = hashlib.sha256(f"{state}|{year}|{C.SEED}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    idx = rng.choice(available, size=n, replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    df.to_parquet(out, index=False)
    _purge_cache()
    return {"state": state, "year": year, "status": "ok", "rows": n,
            "available": available}


def load_window(state: str, year: int) -> pd.DataFrame:
    return pd.read_parquet(window_path(state, year))


def available_cells() -> list[tuple[str, int]]:
    cells = []
    for p in sorted(C.WINDOWS_DIR.glob("*.parquet")):
        state, year = p.stem.split("_")
        cells.append((state, int(year)))
    return cells


def main(argv: list[str]) -> int:
    """CLI: python -m modelwatch.data [year ...]"""
    years = [int(a) for a in argv] if argv else C.YEARS
    log = C.DATA_DIR / "build_log.csv"
    rows = []
    for year in years:
        for state in STATES:
            # the reference cell is drawn larger (see config.REF_N)
            n = C.REF_N if (state == C.REF_STATE and year == C.REF_YEAR) else C.WINDOW_N
            res = build_cell(state, year, n=n)
            rows.append(res)
            flag = {"ok": "+", "cached": ".", "too_small": "s", "error": "!"}[res["status"]]
            print(f"{flag} {state} {year} {res['status']} rows={res['rows']}", flush=True)
            if res["status"] == "error":
                print(f"    {res.get('detail','')}", flush=True)
    pd.DataFrame(rows).to_csv(log, index=False)
    ok = sum(1 for r in rows if r["status"] in ("ok", "cached"))
    print(f"\ncells materialised: {ok}/{len(rows)}  -> {C.WINDOWS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
