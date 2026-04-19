# data/generate_gl_data.py
import pandas as pd
import numpy as np
import sqlite3
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── 1. CHART OF ACCOUNTS ────────────────────────────────────────────────────

accounts = [
    # Assets
    ("1100", "Cash and Central Bank Deposits",       "Asset",    "Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §246"),
    ("1200", "Interbank Loans",                      "Asset",    "Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §246"),
    ("1300", "Customer Loans - Retail",              "Asset",    "Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §253"),
    ("1310", "Customer Loans - Corporate",           "Asset",    "Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §253"),
    ("1400", "Securities Portfolio",                 "Asset",    "Balance Sheet", "IFRS 9 - Fair Value OCI",         "HGB §253"),
    ("1500", "Derivatives - Assets",                 "Asset",    "Balance Sheet", "IFRS 9 - Fair Value P&L",         "HGB §246"),
    ("1600", "Property and Equipment",               "Asset",    "Balance Sheet", "IAS 16",                          "HGB §253"),
    ("1700", "Intangible Assets",                    "Asset",    "Balance Sheet", "IAS 38",                          "HGB §246"),
    ("1800", "Deferred Tax Assets",                  "Asset",    "Balance Sheet", "IAS 12",                          "HGB §274"),
    ("1900", "Other Assets",                         "Asset",    "Balance Sheet", "IAS 39",                          "HGB §246"),

    # Liabilities
    ("2100", "Customer Deposits - Current",          "Liability","Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §246"),
    ("2200", "Customer Deposits - Term",             "Liability","Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §246"),
    ("2300", "Interbank Borrowings",                 "Liability","Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §246"),
    ("2400", "Issued Bonds",                         "Liability","Balance Sheet", "IFRS 9 - Amortised Cost",         "HGB §221"),
    ("2500", "Derivatives - Liabilities",            "Liability","Balance Sheet", "IFRS 9 - Fair Value P&L",         "HGB §246"),
    ("2600", "Tax Liabilities",                      "Liability","Balance Sheet", "IAS 12",                          "HGB §249"),
    ("2700", "Provisions",                           "Liability","Balance Sheet", "IAS 37",                          "HGB §249"),
    ("2800", "Other Liabilities",                    "Liability","Balance Sheet", "IAS 39",                          "HGB §246"),

    # Equity
    ("3100", "Share Capital",                        "Equity",   "Balance Sheet", "IAS 32",                          "HGB §272"),
    ("3200", "Retained Earnings",                    "Equity",   "Balance Sheet", "IAS 1",                           "HGB §272"),
    ("3300", "Other Comprehensive Income",           "Equity",   "Balance Sheet", "IAS 1",                           "HGB §272"),

    # Income
    ("4100", "Interest Income - Retail Loans",       "Income",   "P&L",           "IFRS 9 - Effective Interest",     "HGB §277"),
    ("4110", "Interest Income - Corporate Loans",    "Income",   "P&L",           "IFRS 9 - Effective Interest",     "HGB §277"),
    ("4120", "Interest Income - Securities",         "Income",   "P&L",           "IFRS 9 - Effective Interest",     "HGB §277"),
    ("4200", "Fee and Commission Income",            "Income",   "P&L",           "IFRS 15",                         "HGB §277"),
    ("4300", "Trading Income",                       "Income",   "P&L",           "IFRS 9 - Fair Value P&L",         "HGB §277"),
    ("4400", "Other Operating Income",               "Income",   "P&L",           "IAS 1",                           "HGB §277"),

    # Expenses
    ("5100", "Interest Expense - Deposits",          "Expense",  "P&L",           "IFRS 9 - Effective Interest",     "HGB §278"),
    ("5200", "Interest Expense - Borrowings",        "Expense",  "P&L",           "IFRS 9 - Effective Interest",     "HGB §278"),
    ("5300", "Staff Costs",                          "Expense",  "P&L",           "IAS 19",                          "HGB §278"),
    ("5400", "IT and Infrastructure Costs",          "Expense",  "P&L",           "IAS 16",                          "HGB §278"),
    ("5500", "Loan Loss Provisions",                 "Expense",  "P&L",           "IFRS 9 - ECL",                    "HGB §253"),
    ("5600", "Regulatory and Compliance Costs",      "Expense",  "P&L",           "IAS 37",                          "HGB §278"),
    ("5700", "Depreciation",                         "Expense",  "P&L",           "IAS 16",                          "HGB §253"),
    ("5800", "Other Operating Expenses",             "Expense",  "P&L",           "IAS 1",                           "HGB §278"),
]

gl_accounts = pd.DataFrame(accounts, columns=[
    "gl_number", "account_name", "account_type",
    "reporting_line", "ifrs_category", "hgb_category"
])

# ── 2. MONTHLY BALANCES ──────────────────────────────────────────────────────

entities = ["Solaris SE", "Solaris Bank AG"]
periods  = pd.date_range("2023-01-01", "2024-12-01", freq="MS").strftime("%Y-%m").tolist()

base_amounts = {
    "1100": 850_000_000,   "1200": 1_200_000_000,
    "1300": 3_400_000_000, "1310": 2_100_000_000,
    "1400": 980_000_000,   "1500": 120_000_000,
    "1600": 45_000_000,    "1700": 30_000_000,
    "1800": 12_000_000,    "1900": 67_000_000,
    "2100": 2_800_000_000, "2200": 1_900_000_000,
    "2300": 950_000_000,   "2400": 1_100_000_000,
    "2500": 95_000_000,    "2600": 28_000_000,
    "2700": 55_000_000,    "2800": 42_000_000,
    "3100": 500_000_000,   "3200": 320_000_000,
    "3300": 45_000_000,
    "4100": 12_000_000,    "4110": 8_500_000,
    "4120": 3_200_000,     "4200": 5_800_000,
    "4300": 1_200_000,     "4400": 900_000,
    "5100": 4_200_000,     "5200": 2_800_000,
    "5300": 6_500_000,     "5400": 2_100_000,
    "5500": 1_800_000,     "5600": 1_200_000,
    "5700": 800_000,       "5800": 1_500_000,
}

balances = []
for entity in entities:
    for gl_number, _, account_type, _, _, _ in accounts:
        base = base_amounts.get(gl_number, 1_000_000)
        # entity scaling
        if entity == "Solaris Bank AG":
            base *= 0.6
        # monthly trend with noise
        for i, period in enumerate(periods):
            trend  = 1 + (i * 0.005)               # slight growth over time
            noise  = np.random.normal(1.0, 0.03)    # 3% monthly noise
            amount = round(base * trend * noise, 2)
            balances.append((gl_number, period, amount, "EUR", entity))

gl_balances = pd.DataFrame(balances, columns=[
    "gl_number", "period", "amount", "currency", "entity"
])

# ── 3. TRANSACTIONS ──────────────────────────────────────────────────────────

descriptions = {
    "4100": "Monthly accrual - retail interest income",
    "4110": "Monthly accrual - corporate interest income",
    "4120": "Monthly accrual - securities interest",
    "4200": "Fee income - payment processing",
    "5100": "Monthly accrual - deposit interest expense",
    "5200": "Monthly accrual - borrowing interest expense",
    "5300": "Monthly payroll posting",
    "5400": "IT infrastructure invoice",
    "5500": "ECL provision movement",
    "5600": "Regulatory fee payment",
}

transactions = []
txn_id = 1
for entity in entities:
    for gl_number, _, _, _, _, _ in accounts:
        base = base_amounts.get(gl_number, 1_000_000)
        if entity == "Solaris Bank AG":
            base *= 0.6
        for i, period in enumerate(periods):
            n_txns = random.randint(2, 6)
            for j in range(n_txns):
                day = random.randint(1, 28)
                date = f"{period}-{day:02d}"
                amount = round((base / n_txns) * np.random.normal(1.0, 0.05), 2)
                dc = "C" if gl_number.startswith(("4",)) else "D"
                desc = descriptions.get(
                    gl_number,
                    f"Journal entry - {gl_number}"
                )
                ref = f"JNL-{period}-{txn_id:05d}"
                transactions.append((
                    txn_id, gl_number, date,
                    abs(amount), dc, desc, ref, entity
                ))
                txn_id += 1

gl_transactions = pd.DataFrame(transactions, columns=[
    "transaction_id", "gl_number", "posting_date",
    "amount", "debit_credit", "description", "reference", "entity"
])

# ── 4. SAVE TO SQLITE ────────────────────────────────────────────────────────

from pathlib import Path

# Always create DB inside /data folder safely
BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "gl_database.db"

# create connection
conn = sqlite3.connect(db_path)

# write tables
gl_accounts.to_sql("gl_accounts", conn, if_exists="replace", index=False)
gl_balances.to_sql("gl_balances", conn, if_exists="replace", index=False)
gl_transactions.to_sql("gl_transactions", conn, if_exists="replace", index=False)

conn.close()

print(f"✅ GL database created: {db_path}")
print(f"   gl_accounts:     {len(gl_accounts):>6} rows")
print(f"   gl_balances:     {len(gl_balances):>6} rows")
print(f"   gl_transactions: {len(gl_transactions):>6} rows")

# ── 5. PREVIEW ───────────────────────────────────────────────────────────────

print("\n--- Sample GL Accounts ---")
print(gl_accounts.head(5).to_string(index=False))

print("\n--- Sample Balances (Jan 2024, Solaris SE) ---")
sample = gl_balances[
    (gl_balances.period == "2024-01") &
    (gl_balances.entity == "Solaris SE")
].head(5)
print(sample.to_string(index=False))