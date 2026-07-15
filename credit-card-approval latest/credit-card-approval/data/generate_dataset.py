"""
Credit Card Approval Prediction — Dataset Generator
Generates a realistic synthetic dataset inspired by the UCI Credit Card dataset.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 2000

# ── Demographic features ───────────────────────────────────────────────────
gender          = np.random.choice(['M', 'F'], N, p=[0.44, 0.56])
own_car         = np.random.choice(['Y', 'N'], N, p=[0.34, 0.66])
own_property    = np.random.choice(['Y', 'N'], N, p=[0.72, 0.28])
children_cnt    = np.random.choice([0, 1, 2, 3, 4, 5], N,
                                    p=[0.43, 0.27,0.19, 0.07, 0.03, 0.01])
family_members  = np.random.choice([1, 2, 3, 4, 5], N,
                                    p=[0.15, 0.40, 0.28, 0.12, 0.05])

# ── Income & employment ────────────────────────────────────────────────────
income_type     = np.random.choice(
    ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'],
    N, p=[0.52, 0.23, 0.16, 0.08, 0.01])

annual_income   = np.where(
    income_type == 'Working',
    np.random.normal(180000, 60000, N),
    np.where(income_type == 'Commercial associate',
             np.random.normal(220000, 80000, N),
    np.where(income_type == 'Pensioner',
             np.random.normal(120000, 30000, N),
    np.where(income_type == 'State servant',
             np.random.normal(160000, 40000, N),
             np.random.normal(80000, 20000, N)))))
annual_income   = np.clip(annual_income, 20000, 700000).astype(int)

employment_days = np.where(
    income_type == 'Pensioner',
    np.random.randint(0, 30, N),
    np.random.randint(-15000, -30, N))   # negative = days employed

age_days        = np.random.randint(-25000, -7000, N)   # negative = days from today

education_type  = np.random.choice(
    ['Secondary / secondary special', 'Higher education',
     'Incomplete higher', 'Lower secondary', 'Academic degree'],
    N, p=[0.57, 0.29, 0.09, 0.04, 0.01])

family_status   = np.random.choice(
    ['Married', 'Single / not married', 'Civil marriage',
     'Separated', 'Widow'],
    N, p=[0.64, 0.15, 0.10, 0.07, 0.04])

housing_type    = np.random.choice(
    ['House / apartment', 'With parents', 'Municipal apartment',
     'Rented apartment', 'Office apartment', 'Co-op apartment'],
    N, p=[0.71, 0.12, 0.08, 0.05, 0.02, 0.02])

occupation_type = np.random.choice(
    ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers',
     'Sales staff', 'Cleaning staff', 'Cooking staff', 'Private service staff',
     'Medicine staff', 'Security staff', 'High skill tech staff', 'IT staff', 'Secretaries'],
    N)

# ── Contact / utility ──────────────────────────────────────────────────────
has_work_phone  = np.random.choice([1, 0], N, p=[0.22, 0.78])
has_phone       = np.random.choice([1, 0], N, p=[0.30, 0.70])
has_email       = np.random.choice([1, 0], N, p=[0.09, 0.91])

# ── Target variable: approval status ──────────────────────────────────────
# Derive approval using a realistic scoring logic
score = np.zeros(N)
score += (annual_income > 150000).astype(int) * 2
score += (own_property == 'Y').astype(int) * 1
score += (education_type == 'Higher education').astype(int) * 1.5
score += (education_type == 'Academic degree').astype(int) * 2
score -= (children_cnt >= 3).astype(int) * 1
score += (family_status == 'Married').astype(int) * 0.5
score += (employment_days < -1000).astype(int) * 1.5   # stable employment
score -= (income_type == 'Student').astype(int) * 1
score += np.random.normal(0, 0.8, N)                  # noise

status = (score >= 3.5).astype(int)   # 1 = Approved, 0 = Rejected

# ── Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    'Gender':           gender,
    'Own_Car':          own_car,
    'Own_Property':     own_property,
    'Children_Count':   children_cnt,
    'Annual_Income':    annual_income,
    'Income_Type':      income_type,
    'Education_Type':   education_type,
    'Family_Status':    family_status,
    'Housing_Type':     housing_type,
    'Employment_Days':  employment_days,
    'Age_Days':         age_days,
    'Has_Work_Phone':   has_work_phone,
    'Has_Phone':        has_phone,
    'Has_Email':        has_email,
    'Occupation_Type':  occupation_type,
    'Family_Members':   family_members,
    'Status':           status,
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/credit_card_data.csv', index=False)
print(f"Dataset saved -> data/credit_card_data.csv  ({N} rows)")
print(f"Approval rate : {status.mean()*100:.1f}%")
print(df.head())
