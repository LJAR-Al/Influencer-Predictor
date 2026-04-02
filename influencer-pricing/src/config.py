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

# ── Layer 1: Pre-screening ──

# Countries where APPU is too low to justify spending — auto-skip
LOW_IAP_COUNTRIES = ["PH", "IN", "BD", "PK", "NG", "EG", "ID", "VN", "KE"]
LOW_IAP_COUNTRY_THRESHOLD = 0.30  # Skip if this % or more of audience is from low-IAP countries

# Sub-to-view ratio: flag if views are suspiciously high relative to subscribers
# (e.g. 500k avg views on a 50k sub channel → ratio = 10, very suspicious)
SUSPICIOUS_VIEW_RATIO = 5.0  # views / subs — flag above this

# ── V2 Model: Signup Rate + APPU Weights ──

PLAYBOOK_TABLE = "video_playbook_results"

# Qualitative features from video playbook (assessable pre-campaign)
PLAYBOOK_FEATURES = [
    "content_category",
    "tone_of_speech_gn",
    "fc_creator_enthusiasm_level",
    "hook_category",
    "integration_level",
    "sponsor_placement",
    "audience_product_fit",
]

# Geo tier definitions for APPU weighting
TIER_1_COUNTRIES = ["US", "DE", "GB", "CA", "AU", "NL", "CH", "AT", "SE"]
TIER_2_COUNTRIES = ["FR", "PL", "IT", "KR", "JP", "BE", "ES", "NO", "DK"]

# Signup rate model hyperparameters
SIGNUP_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

# Demographic columns in campaign data
DEMO_AGE_COLS = {
    "calc_pct_age_16_20": "FROM_16_TO_20",
    "calc_pct_age_21_29": "FROM_21_TO_29",
    "calc_pct_age_30_49": "FROM_30_TO_49",
    "calc_pct_age_50plus": "FROM_50",
}
DEMO_GENDER_COLS = {
    "calc_pct_gender_female": "FEMALE",
    "calc_pct_gender_male": "MALE",
}
DEMO_TIER_COLS = ["calc_pct_tier_1", "calc_pct_tier_2", "calc_pct_tier_3", "calc_pct_tier_other"]
