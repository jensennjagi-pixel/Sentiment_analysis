
"""
collect_data.py
----------------
Collects news articles mentioning UK electricity brokers from:
  - NewsAPI (needs free API key from newsapi.org)
  - GDELT (no key needed, good for historical volume)
  - RSS feeds (no key needed)

If no API key is set, falls back to generating realistic sample data
so you can test the full pipeline and Streamlit app immediately.
"""

import os
import time
import random
from typing import Any, Dict, List
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIG — edit this list to track the brokers you care about
# ---------------------------------------------------------------------------
BROKERS = [
    "Utilitywise",
    "Love Energy Savings",
    "Business Energy Comparison",
    "SmartestEnergy",
    "npower Business Solutions",
]

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")  # set via env var, never hardcode a real key

RSS_FEEDS = {
    "utility_week": "https://utilityweek.co.uk/feed/",
    "current_news": "https://www.current-news.co.uk/feed/",
}


# ---------------------------------------------------------------------------
# NewsAPI collection
# ---------------------------------------------------------------------------
def fetch_newsapi(broker: str, days_back: int = 30) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{broker}" AND (energy OR electricity)',
        "language": "en",
        "sortBy": "publishedAt",
        "from": (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "apiKey": NEWSAPI_KEY,
        "pageSize": 50,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "broker": broker,
                "source": a.get("source", {}).get("name", "newsapi"),
                "title": a.get("title", ""),
                "summary": a.get("description", "") or "",
                "published": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            }
            for a in data.get("articles", [])
        ]
    except (requests.RequestException, ValueError) as e:
        print(f"[NewsAPI] error for {broker}: {e}")
        return []


# ---------------------------------------------------------------------------
# GDELT collection (free, no key required)
# ---------------------------------------------------------------------------
def fetch_gdelt(broker: str, max_records: int = 50) -> List[Dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{broker}" energy',
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sourcecountry": "UK",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "broker": broker,
                "source": a.get("domain", "gdelt"),
                "title": a.get("title", ""),
                "summary": "",
                "published": a.get("seendate", ""),
                "url": a.get("url", ""),
            }
            for a in data.get("articles", [])
        ]
    except (requests.RequestException, ValueError) as e:
        print(f"[GDELT] error for {broker}: {e}")
        return []


# ---------------------------------------------------------------------------
# RSS collection (filters feed entries by broker name mention)
# ---------------------------------------------------------------------------
def fetch_rss() -> List[Dict[str, Any]]:
    records = []
    for source, url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            print(f"[RSS] error for {source}: {e}")
            continue
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            for broker in BROKERS:
                if broker.lower() in (title + summary).lower():
                    records.append({
                        "broker": broker,
                        "source": source,
                        "title": title,
                        "summary": summary,
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
    return records


# ---------------------------------------------------------------------------
# Sample data generator — lets you run the whole pipeline with no API keys
# ---------------------------------------------------------------------------
def generate_sample_data(n_per_broker: int = 25) -> pd.DataFrame:
    positive_templates = [
        "{broker} wins customer service award for third year running",
        "{broker} expands into new regional markets with strong growth",
        "{broker} praised by Trustpilot reviewers for transparent pricing",
        "{broker} announces partnership to support small business energy switching",
        "{broker} named accredited broker under new Ofgem code of practice",
    ]
    negative_templates = [
        "{broker} faces Ofgem investigation over mis-selling complaints",
        "Customers report being overcharged by {broker}, watchdog says",
        "{broker} hit with backlash after sudden contract changes",
        "Complaints against {broker} rise sharply amid price volatility",
        "{broker} fined following regulatory breach in sales practices",
    ]
    neutral_templates = [
        "{broker} publishes quarterly market update for business customers",
        "{broker} comments on rising wholesale electricity prices",
        "{broker} to attend upcoming UK energy industry conference",
        "Analysts review {broker}'s position in the business energy market",
        "{broker} updates its broker fee disclosure policy",
    ]

    all_templates = (
        [(t, "positive") for t in positive_templates]
        + [(t, "negative") for t in negative_templates]
        + [(t, "neutral") for t in neutral_templates]
    )

    rows = []
    start_date = datetime.utcnow() - timedelta(days=90)
    for broker in BROKERS:
        for _ in range(n_per_broker):
            template, tone = random.choice(all_templates)
            pub_date = start_date + timedelta(days=random.randint(0, 90))
            rows.append({
                "broker": broker,
                "source": random.choice(["Utility Week", "Current News", "FT Energy", "BBC Business"]),
                "title": template.format(broker=broker),
                "summary": "",
                "published": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "url": "https://example.com/sample-article",
                "_sample_tone": tone,  # only present in sample data, useful for sanity-checking scores
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main collection entry point
# ---------------------------------------------------------------------------
def collect_all(use_live_sources: bool = False) -> pd.DataFrame:
    if not use_live_sources:
        print("Using sample data (set use_live_sources=True once you have API keys configured).")
        return generate_sample_data()

    records = []
    for broker in BROKERS:
        records.extend(fetch_newsapi(broker))
        records.extend(fetch_gdelt(broker))
        time.sleep(1)  # be polite to free-tier rate limits
    records.extend(fetch_rss())

    if not records:
        print("No live records found — falling back to sample data.")
        return generate_sample_data()

    df = pd.DataFrame(records).drop_duplicates(subset=["title", "broker"])
    return df


if __name__ == "__main__":
    df = collect_all(use_live_sources=False)
    df.to_csv("raw_articles.csv", index=False)
    print(f"Saved {len(df)} articles to raw_articles.csv")
