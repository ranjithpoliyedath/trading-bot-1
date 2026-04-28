"""
scripts/probe_alpaca_news.py
------------------------------
Inspects the actual structure of Alpaca's news response so we can
parse it correctly. Prints the raw shape, types, and first article.

Run:
    python scripts/probe_alpaca_news.py
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()


def main():
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )

    print("=" * 60)
    print("Probing Alpaca News API response structure")
    print("=" * 60)

    end   = datetime.utcnow()
    start = end - timedelta(days=2)

    request = NewsRequest(
        symbols="AAPL",
        start=start, end=end,
        limit=3,
        include_content=False,
    )
    response = client.get_news(request)

    print(f"\nResponse type: {type(response).__name__}")
    print(f"Has 'news' attr: {hasattr(response, 'news')}")
    print(f"Has 'data' attr: {hasattr(response, 'data')}")
    print(f"Is iterable: {hasattr(response, '__iter__')}")

    print("\n--- Iteration test ---")
    for i, thing in enumerate(response):
        print(f"  Item {i}: type={type(thing).__name__}, value={thing!r}"[:200])
        if i >= 2:
            break

    print("\n--- Try .data attribute ---")
    if hasattr(response, "data"):
        print(f"  type: {type(response.data).__name__}")
        if isinstance(response.data, dict):
            for key, val in response.data.items():
                print(f"  key={key!r}, val_type={type(val).__name__}, val_len={len(val) if hasattr(val, '__len__') else 'N/A'}")
                if isinstance(val, list) and val:
                    first = val[0]
                    print(f"    First item type: {type(first).__name__}")
                    if hasattr(first, "__dict__"):
                        print(f"    First item attrs: {list(first.__dict__.keys())[:10]}")
                    elif isinstance(first, dict):
                        print(f"    First item keys: {list(first.keys())[:10]}")
                        print(f"    First item: {first}")

    print("\n--- Try .news attribute ---")
    if hasattr(response, "news"):
        news = response.news
        print(f"  type: {type(news).__name__}")
        if isinstance(news, dict):
            for key, val in news.items():
                print(f"  key={key!r}, val_type={type(val).__name__}")
        elif isinstance(news, list):
            print(f"  list length: {len(news)}")
            if news:
                first = news[0]
                print(f"  First type: {type(first).__name__}")
                if hasattr(first, "__dict__"):
                    print(f"  First attrs: {first.__dict__}")


if __name__ == "__main__":
    main()
