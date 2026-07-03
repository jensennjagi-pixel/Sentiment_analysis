"""
app.py
------
Streamlit dashboard for UK electricity broker sentiment analysis.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from collect_data import collect_all, BROKERS
from sentiment_analysis import score_dataframe, aggregate_by_broker_day, broker_leaderboard

st.set_page_config(page_title="UK Electricity Broker Sentiment", layout="wide")

st.title("⚡ UK Electricity Broker Sentiment Tracker")
st.caption(
    "News-based sentiment analysis of UK electricity/business-energy brokers. "
    "Currently running on sample data — swap in a NewsAPI key to go live."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

use_live = st.sidebar.checkbox(
    "Use live data sources (requires NEWSAPI_KEY env var)", value=False
)

selected_brokers = st.sidebar.multiselect(
    "Brokers to display", options=BROKERS, default=BROKERS
)

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()


@st.cache_data(show_spinner="Collecting and scoring articles...")
def load_data(use_live_sources: bool) -> pd.DataFrame:
    raw = collect_all(use_live_sources=use_live_sources)
    scored = score_dataframe(raw)
    return scored


data = load_data(use_live)

if data.empty:
    st.warning("No data available. Try disabling live mode to use sample data.")
    st.stop()

filtered = data[data["broker"].isin(selected_brokers)]

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
st.subheader("Broker Sentiment Leaderboard")

leaderboard = broker_leaderboard(filtered)
leaderboard_display = leaderboard.rename(columns={
    "broker": "Broker",
    "avg_sentiment": "Avg. Sentiment (-1 to 1)",
    "article_count": "Articles",
    "pct_positive": "% Positive",
    "pct_negative": "% Negative",
})

col1, col2 = st.columns([2, 1])

with col1:
    fig_bar = px.bar(
        leaderboard,
        x="broker",
        y="avg_sentiment",
        color="avg_sentiment",
        color_continuous_scale=["#d62728", "#f0f0f0", "#2ca02c"],
        range_color=[-1, 1],
        labels={"broker": "Broker", "avg_sentiment": "Avg. Sentiment"},
        title="Average Sentiment by Broker",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.dataframe(
        leaderboard_display.style.format({
            "Avg. Sentiment (-1 to 1)": "{:.2f}",
            "% Positive": "{:.0f}%",
            "% Negative": "{:.0f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Sentiment over time
# ---------------------------------------------------------------------------
st.subheader("Sentiment Trend Over Time")

daily = aggregate_by_broker_day(filtered)

if not daily.empty:
    fig_line = px.line(
        daily,
        x="date",
        y="avg_sentiment",
        color="broker",
        markers=True,
        labels={"date": "Date", "avg_sentiment": "Avg. Daily Sentiment", "broker": "Broker"},
    )
    fig_line.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Not enough dated articles to show a trend yet.")

# ---------------------------------------------------------------------------
# Sentiment distribution
# ---------------------------------------------------------------------------
st.subheader("Sentiment Mix by Broker")

mix = (
    filtered.groupby(["broker", "sentiment_label"])
    .size()
    .reset_index(name="count")
)
fig_mix = px.bar(
    mix,
    x="broker",
    y="count",
    color="sentiment_label",
    color_discrete_map={"positive": "#2ca02c", "neutral": "#a0a0a0", "negative": "#d62728"},
    labels={"broker": "Broker", "count": "Article Count", "sentiment_label": "Sentiment"},
    barmode="stack",
)
st.plotly_chart(fig_mix, use_container_width=True)

# ---------------------------------------------------------------------------
# Recent headlines table
# ---------------------------------------------------------------------------
st.subheader("Recent Headlines")

display_cols = ["published", "broker", "source", "title", "compound_score", "sentiment_label"]
recent = filtered.sort_values("published", ascending=False)[display_cols].head(50)
recent = recent.rename(columns={
    "published": "Date",
    "broker": "Broker",
    "source": "Source",
    "title": "Headline",
    "compound_score": "Score",
    "sentiment_label": "Sentiment",
})

def highlight_sentiment(val):
    if val == "positive":
        return "color: #2ca02c"
    elif val == "negative":
        return "color: #d62728"
    return "color: #808080"

st.dataframe(
    recent.style.map(highlight_sentiment, subset=["Sentiment"]).format({"Score": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Sentiment scored with VADER + a custom lexicon tuned for energy-broker language "
    "(mis-selling, overcharged, award, accreditation, etc). "
    "Thresholds: positive ≥ 0.05, negative ≤ -0.05, else neutral."
)
