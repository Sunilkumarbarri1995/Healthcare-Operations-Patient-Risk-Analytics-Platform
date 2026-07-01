import pandas as pd
import numpy as np

# ── File paths ─────────────────────────────────────────────────────────────
DATA_DIR = "/Users/sunilkumarbarri/Desktop/Learning/HEALTHCARE PROVIDER FRAUD DETECTION ANALYSIS Dataset"

labels      = pd.read_csv(f"{DATA_DIR}/Train-1542865627584.csv")
inpatient   = pd.read_csv(f"{DATA_DIR}/Train_Inpatientdata-1542865627584.csv", low_memory=False)
outpatient  = pd.read_csv(f"{DATA_DIR}/Train_Outpatientdata-1542865627584.csv", low_memory=False)
beneficiary = pd.read_csv(f"{DATA_DIR}/Train_Beneficiarydata-1542865627584.csv", low_memory=False)

print("=" * 55)
print("STEP 1 — FILE SHAPES")
print("=" * 55)
print(f"  Labels:      {labels.shape}")
print(f"  Inpatient:   {inpatient.shape}")
print(f"  Outpatient:  {outpatient.shape}")
print(f"  Beneficiary: {beneficiary.shape}")

# ── Step 2: Basic integrity ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — MISSING VALUES & DUPLICATES")
print("=" * 55)
for name, df in [("Labels", labels), ("Inpatient", inpatient),
                 ("Outpatient", outpatient), ("Beneficiary", beneficiary)]:
    nulls = df.isnull().sum().sum()
    dups  = df.duplicated().sum()
    print(f"  {name:12s} → nulls: {nulls:,}   duplicates: {dups:,}")

# ── Step 3: Target label distribution ─────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — TARGET LABEL (FRAUD DISTRIBUTION)")
print("=" * 55)
print(labels['PotentialFraud'].value_counts().to_string())
fraud_rate = labels['PotentialFraud'].eq('Yes').mean()
print(f"\n  Fraud rate: {fraud_rate:.2%}  ← expect ~10%")

# ── Step 4: Join key integrity ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — PROVIDER JOIN KEY INTEGRITY")
print("=" * 55)
label_providers     = set(labels['Provider'].unique())
inpatient_providers = set(inpatient['Provider'].unique())
outpatient_providers= set(outpatient['Provider'].unique())

print(f"  Providers in labels:     {len(label_providers):,}")
print(f"  Providers in inpatient:  {len(inpatient_providers):,}")
print(f"  Providers in outpatient: {len(outpatient_providers):,}")

orphan_in  = inpatient_providers  - label_providers
orphan_out = outpatient_providers - label_providers
print(f"\n  Inpatient providers NOT in labels:  {len(orphan_in)}  ← expect 0")
print(f"  Outpatient providers NOT in labels: {len(orphan_out)}  ← expect 0")

# ── Step 5: Date & numeric sanity ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5 — DATE & NUMERIC SANITY")
print("=" * 55)
inpatient['AdmissionDt'] = pd.to_datetime(inpatient['AdmissionDt'])
inpatient['DischargeDt'] = pd.to_datetime(inpatient['DischargeDt'])
inpatient['HospDays']    = (inpatient['DischargeDt'] - inpatient['AdmissionDt']).dt.days

neg_days = (inpatient['HospDays'] < 0).sum()
max_days = inpatient['HospDays'].max()
neg_reimb = (inpatient['InscClaimAmtReimbursed'] < 0).sum()
zero_reimb = (inpatient['InscClaimAmtReimbursed'] == 0).sum()

print(f"  Negative hospitalization durations: {neg_days}   ← expect 0")
print(f"  Max hospitalization days:           {max_days}")
print(f"  Negative reimbursement amounts:     {neg_reimb}   ← expect 0")
print(f"  Zero reimbursement rows:            {zero_reimb}")

# ── Step 6: Beneficiary demographics check ────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6 — BENEFICIARY DEMOGRAPHICS")
print("=" * 55)
print(f"  Unique beneficiaries: {beneficiary['BeneID'].nunique():,}")
print(f"  Gender distribution:\n{beneficiary['Gender'].value_counts().to_string()}")
print(f"  Chronic conditions columns: {[c for c in beneficiary.columns if 'ChronicCond' in c]}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)
checks = {
    "Files loaded correctly":        True,
    "No orphaned inpatient records":  len(orphan_in) == 0,
    "No orphaned outpatient records": len(orphan_out) == 0,
    "No negative stay durations":     neg_days == 0,
    "No negative reimbursements":     neg_reimb == 0,
    "Fraud rate ~10%":               0.05 < fraud_rate < 0.20,
}
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")
