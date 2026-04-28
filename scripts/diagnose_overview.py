"""
scripts/diagnose_overview.py
-----------------------------
Diagnostic for empty Market Overview panels.

Reports:
  - How many feature files exist
  - Volume ratio distribution and current top movers
  - Sentiment data presence (zero vs real values)
  - Latest dates in feature data

Run:
    python scripts/diagnose_overview.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def main():
    files = sorted(DATA_DIR.glob("*_features.parquet"))
    print(f"Total feature files: {len(files)}")
    if not files:
        print("No feature files — run `python -m bot.pipeline` first.")
        return

    # ── Sample stats per symbol ──────────────────────────────────────────────
    print()
    print(f'{"symbol":<8} {"rows":<7} {"latest_date":<12} {"max_vr":<8} '
          f'{"sent_nonzero":<14} {"latest_sent":<12}')
    print("-" * 70)

    for fp in files[:10]:
        sym = fp.stem.replace("_features", "")
        df  = pd.read_parquet(fp)

        vr_max       = df["volume_ratio"].max() if "volume_ratio" in df.columns else 0
        sent_col     = df.get("combined_sentiment", pd.Series([0]))
        sent_nonzero = (sent_col.abs() > 0).sum()
        sent_last    = sent_col.iloc[-1] if len(sent_col) else 0
        latest_date  = str(df.index[-1])[:10] if len(df) else "—"

        print(f"{sym:<8} {len(df):<7} {latest_date:<12} {vr_max:<8.2f} "
              f"{sent_nonzero:<14} {sent_last:<12.3f}")

    # ── Today's top movers across all symbols ────────────────────────────────
    print()
    print("Top 10 symbols by LATEST volume_ratio:")
    movers = []
    for fp in files:
        sym = fp.stem.replace("_features", "")
        df  = pd.read_parquet(fp)
        if "volume_ratio" in df.columns and len(df):
            vr = df["volume_ratio"].iloc[-1]
            if pd.notna(vr):
                movers.append((sym, float(vr)))

    movers.sort(key=lambda x: x[1], reverse=True)
    for sym, vr in movers[:10]:
        print(f"  {sym:<6}  {vr:.2f}x")

    # ── Sentiment summary across all symbols ─────────────────────────────────
    print()
    print("Sentiment summary across all feature files:")
    total_with_sent = 0
    total_nonzero   = 0
    for fp in files:
        df = pd.read_parquet(fp)
        if "combined_sentiment" in df.columns:
            total_with_sent += 1
            if (df["combined_sentiment"].abs() > 0).any():
                total_nonzero += 1
    print(f"  Files with combined_sentiment column: {total_with_sent} / {len(files)}")
    print(f"  Files with at least one non-zero score: {total_nonzero} / {len(files)}")


if __name__ == "__main__":
    main()
