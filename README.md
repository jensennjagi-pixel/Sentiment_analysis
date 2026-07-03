# UK Electricity Broker Sentiment Tracker

## What's in here
- `collect_data.py` — collects broker news from NewsAPI, GDELT, and RSS feeds. Falls back to realistic sample data if no API key is set, so you can run everything immediately.
- `sentiment_analysis.py` — cleans text and scores sentiment with VADER + a custom lexicon tuned for energy-broker language (mis-selling, overcharged, award, accreditation, etc).
- `app.py` — the Streamlit dashboard: leaderboard, sentiment trend over time, sentiment mix, and a recent headlines table.
- `requirements.txt` — dependencies.

## Quick start (with sample data — no setup needed)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This runs immediately using generated sample data so you can see exactly how the dashboard behaves.

## Going live with real data

1. Get a free API key from https://newsapi.org (100 requests/day on the free tier).
2. Set it as an environment variable before running:

   ```bash
   export NEWSAPI_KEY="your_key_here"
   streamlit run app.py
   ```
3. In the app sidebar, tick **"Use live data sources"**.

GDELT and RSS sources don't require any key and will run automatically.

## Editing which brokers you track

Open `collect_data.py` and edit the `BROKERS` list near the top:

```python
BROKERS = [
    "Utilitywise",
    "Love Energy Savings",
    "Business Energy Comparison",
    "SmartestEnergy",
    "npower Business Solutions",
]
```

## Tuning sentiment

Open `sentiment_analysis.py` and edit `CUSTOM_LEXICON` — add words you notice appearing often in broker news that VADER scores incorrectly (each word gets a score from roughly -4 to +4).

## Adding electricity price correlation (optional next step)

To check whether broker sentiment dips line up with wholesale price spikes:
1. Register for a free Elexon BMRS API key.
2. Pull daily day-ahead prices into a DataFrame with `date` and `price` columns.
3. Merge on date with the output of `aggregate_by_broker_day()` in `sentiment_analysis.py`.
4. Use `statsmodels` (`grangercausalitytests`) to test for lead/lag relationships.

Happy to help build that piece out once the core dashboard is working for you.
