"""Configuration for the influencer pricing pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ygsjtngtidtmzojugiji.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

TABLE_NAME = "creatorshub_campaigns_data"

# IAP metric: purchases by d7 users measured at day 14 after publish
IAP_COL = "d7_user_purchases_by_d14_publish_date"

# Minimum cost to include a training row
MIN_COST = 1.0

# Minimum expected views to include (filters out entries with no reach data)
MIN_EXPECTED_VIEWS = 1

# Profitability rule: d7 users' IAP at day 14 must be >= this fraction of cost
# 0.10 = campaign is profitable if IAP >= 10% of price paid
# Therefore: max_price = predicted_iap / PROFITABILITY_THRESHOLD
PROFITABILITY_THRESHOLD = 0.10

# Pre-campaign features (available before signing a creator)
PRE_CAMPAIGN_NUMERIC = [
    "expected_views",           # Average view count / reach
    "expected_cpm",             # Pre-campaign CPM valuation
    "demographics_female_pct",
    "demographics_male_pct",
    "demographics_other_pct",
]

PRE_CAMPAIGN_CATEGORICAL = [
    "youtube_category_name",
    "demographics_main_country",
]

# Only train/score on YouTube data — Instagram has different CPM dynamics
PLATFORM_FILTER = "Youtube"

# Quantile levels for price ranges
# These are applied to the profitable campaign CPM distribution
QUANTILES = {
    "conservative": 0.25,  # P25 of profitable CPMs
    "moderate":     0.50,  # Median of profitable CPMs
    "aggressive":   0.75,  # P75 of profitable CPMs
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Classifier threshold: minimum P(conversion) to consider a creator
# Lowered from 0.5 to 0.3 to reduce missed profitable deals
# (backtest: catches 1 more profitable campaign with minimal risk)
CLASSIFIER_THRESHOLD = 0.3

# Pricing strategy (backtested):
# - New creators: moderate benchmark (P50 CPM)
# - Rebookings: aggressive benchmark (P75 CPM) — more confidence from real data
DEFAULT_NEW_LEVEL = "moderate"
DEFAULT_REBOOKING_LEVEL = "aggressive"
