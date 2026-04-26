"""
scripts/wire_overview_page.py
------------------------------
One-time setup: adds the Market Overview page to the dashboard app
and ensures the index/sector ETFs are part of the universe pipeline.

Run once after installing the step-6 zip:
    python scripts/wire_overview_page.py
"""

from pathlib import Path

ROOT      = Path(__file__).parent.parent
APP_PATH  = ROOT / "dashboard" / "app.py"
CFG_PATH  = ROOT / "bot" / "config.py"

# 1. Wire the overview page into app.py routing
def patch_app():
    if not APP_PATH.exists():
        print(f"⚠ {APP_PATH} not found — skipping app.py patch.")
        return

    content = APP_PATH.read_text()

    if "from dashboard.pages import market_overview" in content:
        print("✓ app.py already wired.")
        return

    # Add import at the top (after other dashboard.pages imports)
    if "from dashboard.pages import overview" in content:
        content = content.replace(
            "from dashboard.pages import overview",
            "from dashboard.pages import overview, market_overview",
            1,
        )
    else:
        marker = "from dashboard.components.global_controls import render_topbar"
        content = content.replace(
            marker,
            "from dashboard.pages import market_overview\n" + marker,
            1,
        )

    # Replace overview.layout(...) with market_overview.layout(...) in the router
    content = content.replace(
        "return overview.layout(account, model, symbol)",
        "return market_overview.layout(account, model, symbol)",
    )

    APP_PATH.write_text(content)
    print("✓ app.py wired to use market_overview as default page.")


# 2. Add INDEX_ETFS + SECTOR_ETFS to config so pipeline fetches them
def patch_config():
    if not CFG_PATH.exists():
        print(f"⚠ {CFG_PATH} not found — skipping config patch.")
        return

    content = CFG_PATH.read_text()
    if "INDEX_ETFS" in content:
        print("✓ config.py already has ETF symbols.")
        return

    addition = '''

# ── Index & Sector ETFs (always included in pipeline regardless of universe) ──
INDEX_ETFS  = ["SPY", "QQQ", "DIA", "IWM", "VTI"]
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
'''
    content = content.rstrip() + addition + "\n"
    CFG_PATH.write_text(content)
    print("✓ config.py updated with INDEX_ETFS + SECTOR_ETFS.")


if __name__ == "__main__":
    patch_app()
    patch_config()
    print("\nDone. Run pipeline to fetch ETF data:")
    print("  python -m bot.pipeline --symbols SPY,QQQ,DIA,IWM,VTI,XLK,XLV,XLF,XLE,XLY,XLP,XLI,XLU,XLB,XLRE,XLC")
    print("\nThen launch dashboard:")
    print("  python -m dashboard.app")
