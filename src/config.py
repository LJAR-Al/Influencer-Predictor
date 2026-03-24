"""Configuration for the influencer pricing pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ygsjtngtidtmzojugiji.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

TABLE_NAME = "creatorshub_campaigns_data"

# Revenue metric: IAP (in-app purchases) data
REVENUE_COL = "d7_user_purchases_by_d14_publish_date"

# Minimum cost to include a training row
MIN_COST = 1.0

# Minimum expected views to include (filters out entries with no reach data)
MIN_EXPECTED_VIEWS = 1

# Profitability rule: d7 IAP at day 14 must be >= this fraction of cost
# 0.10 = campaign is profitable if IAP revenue >= 10% of price paid
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
    "posting_platform",
    "youtube_category_name",
    "demographics_main_country",
]

# Quantile levels for price ranges (tighter bands)
QUANTILES = {
    "conservative": 0.30,  # 70% confident revenue will be at least this
    "moderate":     0.50,  # 50/50
    "aggressive":   0.70,  # 30% confident, higher upside
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
