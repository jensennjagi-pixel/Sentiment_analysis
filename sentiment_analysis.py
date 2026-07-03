"""
sentiment_analysis.py
----------------------
Cleans article text and scores sentiment using VADER, adjusted with a
custom lexicon for energy-broker-specific language that generic
sentiment models tend to miss or under-weight.
"""

import re
from typing import Any
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import ssl

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Custom lexicon additions — tune these based on what you observe.
# VADER scores range roughly -4 (very negative) to +4 (very positive).
# ---------------------------------------------------------------------------
CUSTOM_LEXICON = {
    "mis-selling": -3.0,
    "misselling": -3.0,
    "overcharged": -2.5,
    "overcharge": -2.5,
    "investigation": -1.8,
    "backlash": -2.2,
    "complaint": -1.5,
    "complaints": -1.5,
    "fined": -2.5,
    "fine": -1.5,
    "collapse": -3.0,
    "collapsed": -3.0,
    "breach": -2.0,
    "watchdog": -1.0,
    "award": 2.0,
    "awards": 2.0,
    "accredited": 1.8,
    "accreditation": 1.8,
    "expansion": 1.5,
    "expands": 1.5,
    "partnership": 1.2,
    "trustpilot": 0.8,  # context-dependent; usually appears in positive review coverage
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9£%.,'\-\s]", "", text)
    return text.strip()


def build_analyzer() -> Any:
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        class FallbackAnalyzer:
            def __init__(self) -> None:
                self.lexicon = dict(CUSTOM_LEXICON)

            def polarity_scores(self, text: str) -> dict:
                words = re.findall(r"[a-zA-Z']+", text.lower())
                score = sum(self.lexicon.get(word, 0.0) for word in words)
                capped = max(min(score / 10.0, 1.0), -1.0)
                if capped >= 0.05:
                    sentiment = "positive"
                elif capped <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                return {
                    "neg": max(0.0, -capped),
                    "neu": 1.0 - abs(capped),
                    "pos": max(0.0, capped),
                    "compound": capped,
                    "sentiment": sentiment,
                }

        return FallbackAnalyzer()

    sia.lexicon.update(CUSTOM_LEXICON)
    return sia


def classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with 'title' and 'summary' columns.
    Adds: clean_text, compound_score, sentiment_label
    """
    sia = build_analyzer()

    df = df.copy()
    df["clean_text"] = (
        df["title"].fillna("").apply(clean_text)
        + ". "
        + df["summary"].fillna("").apply(clean_text)
    )
    df["compound_score"] = df["clean_text"].apply(
        lambda t: sia.polarity_scores(t)["compound"] if t.strip() else 0.0
    )
    df["sentiment_label"] = df["compound_score"].apply(classify)

    # parse dates safely
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)

    return df


def aggregate_by_broker_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily average sentiment + article volume per broker.
    """
    df = df.dropna(subset=["published"])
    df["date"] = df["published"].dt.date

    agg = (
        df.groupby(["broker", "date"])
        .agg(
            avg_sentiment=("compound_score", "mean"),
            article_count=("compound_score", "count"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
        )
        .reset_index()
    )
    return agg


def broker_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall summary per broker across the full collected period.
    """
    summary = (
        df.groupby("broker")
        .agg(
            avg_sentiment=("compound_score", "mean"),
            article_count=("compound_score", "count"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
        )
        .reset_index()
        .sort_values("avg_sentiment", ascending=False)
    )
    return summary


if __name__ == "__main__":
    raw = pd.read_csv("raw_articles.csv")
    scored = score_dataframe(raw)
    scored.to_csv("scored_articles.csv", index=False)
    print(f"Scored {len(scored)} articles -> scored_articles.csv")
    print(broker_leaderboard(scored))
