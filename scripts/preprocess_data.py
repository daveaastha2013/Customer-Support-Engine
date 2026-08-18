import os
import json
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standardized Categories
CATEGORY_MAPPING = {
    'PAYMENT': 'payment',
    'ACCOUNT': 'account',
    'DELIVERY': 'delivery',
    'REFUND': 'refund',
    'TECHNICAL': 'technical',
    'SUBSCRIPTION': 'subscription'
}

URGENT_KEYWORDS = ['twice', 'urgent', 'immediately', 'cancel', 'lawyer', 'failed', 'shattered', 'crushed', 'contacted', 'stolen', 'pending']

def compute_sentiment(text):
    """Computes sentiment polarity and subjectivity using TextBlob."""
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def derive_proxy_escalation(row):
    """
    EXPERIMENTAL HEURISTIC PROXY ESCALATION LABEL:
    This is an experimental proxy model logic. It DOES NOT represent real customer escalation outcomes.
    Formula:
    High risk (1) if sentiment is negative AND (message length > 120 OR contact frequency >= 2 OR urgent keyword present).
    """
    polarity = row['sentiment_polarity']
    length = row['message_length']
    urgent_count = row['urgency_keywords']
    interactions = row['interaction_count']
    
    # Heuristic scoring rule
    is_negative = polarity < -0.05
    has_urgency = urgent_count > 0 or interactions >= 2
    is_long = length > 120
    
    if is_negative and (has_urgency or is_long):
        return 1
    elif urgent_count >= 2:
        return 1
    return 0

def create_knowledge_base_docs():
    """Generates structured FAQ & Policy markdown documents into data/knowledge_base/."""
    settings.KB_DIR.mkdir(parents=True, exist_ok=True)
    
    kb_docs = {
        "payment_policy.md": """# Payment & Billing Guidelines

## Payment Processing & Pending Status
Orders placed with credit card, debit card, or PayPal usually reflect in your dashboard within 15 minutes. 
If your payment went through but the order status displays 'Pending' for over 30 minutes, it indicates an automated payment gateway sync verification. 
Do not attempt a second payment as you may be double-billed. The automated system syncs transactions within 2 hours.

## Failed Transactions & Holds
If a payment fails during checkout but your bank account reflects a deduction, this is a temporary authorization hold. 
The funds are automatically released by your card issuing bank within 3 to 5 business days.

## Tax & VAT Invoices
Tax and VAT invoices can be downloaded directly from your Account Settings > Billing History > Download Invoice PDF. 
If corporate details need to be amended on past invoices, submit a ticket under Account & Billing.
""",

        "account_security.md": """# Account & Authentication FAQ

## Password Reset Procedures
To reset your account password, click 'Forgot Password' on the login screen and enter your registered email.
Password reset email links expire after 30 minutes for security reasons. If the link is not received, check Spam/Junk folders.

## Two-Factor Authentication (2FA) Recovery
If you lose your primary 2FA authentication device, use one of the 8-digit emergency recovery codes provided during initial setup.
If emergency recovery codes are unavailable, account verification via photo ID and billing statement confirmation is required.

## Account Deletion & GDPR Data Rights
Account deletion permanently removes personal information within 30 days pursuant to GDPR guidelines.
Navigate to Profile Settings > Privacy > Delete My Account. Active subscriptions must be cancelled prior to deletion.
""",

        "shipping_returns.md": """# Shipping, Delivery & Damaged Goods

## Package Tracking & Shipment Delays
Tracking details update within 24 hours of package dispatch. 
Carrier tracking status may occasionally remain unchanged for up to 4 business days while packages are in transit between sorting hubs or undergoing customs processing.

## Address Modifications Post-Purchase
Shipping address changes are permissible within 2 hours of placing an order if the status is 'Processing'.
Once an order reaches 'Dispatched' status, address modifications cannot be completed.

## Damaged & Shattered Items Protocol
If your package arrives crushed or items inside are damaged/shattered upon delivery:
1. Retain original packaging and damaged contents.
2. Submit clear photos of the shipping box and item to Support.
3. A free prepaid replacement shipment will be issued within 1 business day.
""",

        "refund_policy.md": """# Refund & Returns Policy

## 30-Day Money-Back Guarantee
All digital products and hardware items purchased directly from our official portal carry a 30-day money-back guarantee.
To initiate a refund, navigate to Orders > Request Refund, download the prepaid shipping label, and ship the item back.

## Refund Processing Timeline
Refunds are processed within 2-3 business days of the returned item reaching our distribution warehouse.
Once issued, financial institutions take 5 to 10 business days to credit the refunded funds to your original payment method.
""",

        "technical_troubleshooting.md": """# Technical Support & API Integration

## Application Crashes & Memory Errors
For persistent mobile application crashes or UI freezes, perform the following troubleshooting steps:
1. Ensure your device OS is updated to the latest supported version.
2. Clear mobile application cache under Settings > Apps > Storage > Clear Cache.
3. Reinstall the latest application build from the App Store / Google Play Store.

## API Rate Limit (HTTP 429) Handling
HTTP 429 'Too Many Requests' errors occur when exceeding API rate quotas.
Inspect response headers for `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`.
Implement exponential backoff algorithms with jitter to prevent rate limit throttling.
""",

        "subscription_management.md": """# Subscriptions & Plan Tiers

## Subscription Cancellation & Auto-Renewal
Auto-renewing subscriptions can be cancelled anytime under Account > Subscriptions > Turn Off Auto-Renew.
Cancelling auto-renewal prevents future billing while retaining active feature access through the end of your current paid billing period.

## Plan Upgrades & Pro-Rated Billing
Upgrading from Basic to Pro or Team tiers occurs immediately upon selection.
Your account will be credited for unused time on the previous tier and billed a pro-rated difference for the remaining billing cycle.
"""
    }

    for filename, content in kb_docs.items():
        doc_path = settings.KB_DIR / filename
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
    logger.info(f"Generated {len(kb_docs)} knowledge base documentation files in {settings.KB_DIR}")

def preprocess_pipeline():
    """Main data preprocessing pipeline."""
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = settings.RAW_DATA_DIR / "bitext_raw.csv"
    
    if not raw_path.exists():
        logger.warning("Raw data file missing. Triggering download script...")
        from download_data import download_bitext_dataset
        download_bitext_dataset()
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw dataset with {len(df)} rows.")
    
    # Ensure text column is clean
    if 'instruction' in df.columns:
        text_col = 'instruction'
    elif 'utterance' in df.columns:
        text_col = 'utterance'
    else:
        text_col = df.columns[0]
        
    df['text'] = df[text_col].astype(str).str.strip()
    df['response'] = df['response'].astype(str).str.strip()
    
    # Standardize category labels
    df['category'] = df['category'].astype(str).str.upper().map(lambda c: CATEGORY_MAPPING.get(c, 'payment'))
    
    # Feature Engineering
    sentiments = [compute_sentiment(t) for t in df['text']]
    df['sentiment_polarity'] = [s[0] for s in sentiments]
    df['sentiment_subjectivity'] = [s[1] for s in sentiments]
    df['message_length'] = df['text'].apply(len)
    
    def count_urgency(t):
        t_lower = t.lower()
        return sum(1 for kw in URGENT_KEYWORDS if kw in t_lower)
    
    df['urgency_keywords'] = df['text'].apply(count_urgency)
    # Simulate contact count (e.g. "contacted support twice" -> 2 interaction count, else 1)
    df['interaction_count'] = df['text'].apply(lambda t: 3 if 'twice' in t.lower() or 'contacted' in t.lower() else 1)
    
    # Derive Experimental Proxy Escalation Label
    df['is_escalated_proxy'] = df.apply(derive_proxy_escalation, axis=1)
    
    # Create unique ticket IDs
    df['ticket_id'] = [f"TICK-{i+1001:05d}" for i in range(len(df))]
    
    # 1. Save Historical Support Tickets (REAL Bitext Records)
    historical_tickets = []
    for idx, row in df.iterrows():
        historical_tickets.append({
            "ticket_id": row['ticket_id'],
            "query": row['text'],
            "resolution": row['response'],
            "category": row['category'],
            "source_type": "REAL_BITEXT_RECORD"
        })
    
    hist_path = settings.PROCESSED_DATA_DIR / "historical_tickets.json"
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump(historical_tickets, f, indent=2)
    logger.info(f"Saved {len(historical_tickets)} real Bitext historical ticket resolution pairs to {hist_path}")
    
    # 2. Save Processed Tickets Dataset
    processed_csv = settings.PROCESSED_DATA_DIR / "tickets_processed.csv"
    df.to_csv(processed_csv, index=False)
    logger.info(f"Saved processed dataset to {processed_csv}")
    
    # 3. Train / Validation / Test Splits (70% Train, 15% Val, 15% Test)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=settings.SEED, stratify=df['category'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=settings.SEED, stratify=test_df['category'])
    
    train_df.to_csv(settings.PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(settings.PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(settings.PROCESSED_DATA_DIR / "test.csv", index=False)
    
    logger.info(f"Dataset splits created -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 4. Generate Knowledge Base Documents
    create_knowledge_base_docs()
    
    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_pipeline()
