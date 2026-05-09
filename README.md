# Risk Segmentation & Diagnostic Strategy — CAT 1

**Course:** DSA 8304 — Risk Management Analytics  
**Institution:** Strathmore University  
**Dataset:** MSMEs from Kenya, Uganda, DRC, and Ghana  

---

## Project Overview

A microfinance institution expanding into African markets needs to understand the risk profiles of Micro, Small, and Medium Enterprises (MSMEs), identify key drivers of high-risk businesses, and develop tailored risk management strategies for different customer segments.

This project uses unsupervised learning for customer segmentation, decision trees for root cause analysis, and translates analytical outputs into actionable lending strategies.

---

## Part 1: Customer Segmentation (Complete)

### What Was Done

#### 1. Data Cleaning
- **Type standardisation:** Integer columns cast to nullable `Int64`, financial columns to `float64`, and categorical strings normalised (smart quotes, inconsistent labels).
- **Missing value treatment:** Four-step imputation strategy informed by a missingness correlation analysis:
  - Country-aware "Not Surveyed" coding for questions never administered in certain countries.
  - Correlation-driven conditional imputation (e.g., cash flow problems inferred from liquidity constraints).
  - Country-level mode/median imputation for remaining nulls.
- **Outlier detection:** IQR-based winsorisation on financial columns (`gross_income`, `operating_expense`, `business_sales`).
- **Log transformation:** `np.log1p` applied to financial columns to reduce right skew.

#### 2. Feature Engineering
Nine domain-relevant features were created:

| Feature | Description |
|---|---|
| `profit_margin` | log(income) - log(expenses) — financial health indicator |
| `revenue_efficiency` | Income-to-sales ratio — operational efficiency |
| `digital_access_score` | Count of digital finance tools (mobile wallet, internet banking) |
| `insurance_gap` | Has inventory risk but no insurance — unmitigated exposure |
| `extends_credit` | Lends to customers — secondary credit risk |
| `financial_stress_score` (0-5) | Composite of 5 distress signals (cash flow, liquidity, informal lending, funding barriers) |
| `formality_score` (0-5) | Composite of 5 formal access indicators (insurance, banking, cards, tax compliance) |
| `business_age_years` | Business age in years |
| `age_group` | Owner age binned into ordinal bands |

#### 3. Exploratory Data Analysis
Three-layer EDA conducted with "Not Surveyed" values excluded:
- **Univariate:** Distributions of numeric and categorical risk variables.
- **Bivariate:** Engineered scores by country (violin plots), key risk indicators (stacked bars), stress vs. profit margin scatter plots.
- **Multivariate:** Feature correlation matrix (identified `log_gross_income` / `log_business_sales` collinearity at r=0.97), country risk profile heatmap.

#### 4. Clustering
- **Algorithm:** K-Means (k=4), validated with elbow method and silhouette scores (peak at 0.2916).
- **Features used:** `log_gross_income`, `profit_margin`, `financial_stress_score`, `formality_score`, `years_in_operation`, `customer_age`, `survey_coverage`.
- **Validation:** Agglomerative clustering comparison (ARI = 0.4462 — moderate agreement confirming structure).
- **Visualisation:** PCA 2D projection of clusters.

#### 5. Segments Identified

| Tier | Cluster | Name | Size | Risk Level | Key Characteristics |
|---|---|---|---|---|---|
| 1 | Cluster 1 | **Stable Formal** | 22% | Lowest | Best margins (0.45), highest formality (2.03), experienced owners. DRC-dominated (74%). |
| 2 | Cluster 0 | **Large Informal** | 42% | Moderate | Largest income but formality score 0.28. Ghana (46%) & Kenya (52%). Profitable but no formal buffer. |
| 3 | Cluster 3 | **Established Veterans** | 11% | Moderate-High | Oldest businesses (22 yrs), slightly loss-making. Only cross-country segment. Succession risk. |
| 4 | Cluster 2 | **High Risk Informal** | 25% | Highest | Worst margins (-2.13), highest stress (4.39), near-zero formality. 98% Uganda, 67% female-owned. |

#### Key Findings
- Three of four clusters are country-dominated, reflecting national economic structures.
- Cluster 2 (High Risk Informal) is 67% female-owned — the most vulnerable segment disproportionately affects women-owned businesses, making any intervention a de facto women's financial inclusion programme.
- Only Cluster 3 (Established Veterans) cuts meaningfully across all four countries.

---

## Part 2: Root Cause Analysis — Orange Data Mining

Part 2 uses the **Orange Data Mining** tool to build decision tree classifiers that identify what drives businesses into each risk segment. The goal is to answer: *"What characteristics make a business end up in the High Risk cluster vs. the Stable Formal cluster?"*

**Deliverables:**
- Decision tree classifiers for each segment
- Top 5 risk drivers per segment
- Business-language interpretation of what each driver means for lending

---

### Step 0 — Prepare the Data

Before opening Orange, export the clustered dataset from the notebook. Add and run this cell at the very end of the notebook:

```python
# Export clustered data for Orange
orange_cols = [
    'Nationality', 'Gender', 'customer_age', 'years_in_operation',
    'has_insurance', 'active_loan_holder', 'has_internet_banking',
    'has_debit_card', 'credit_card_ownership', 'compliance_income_tax',
    'current_problem_cash_flow', 'liquidity_constraint',
    'uses_informal_lender', 'uses_friends_family_savings',
    'funding_access_barrier', 'mobile_wallet_access',
    'extends_customer_credit', 'inventory_shrinkage_risk',
    'financial_tracking',
    'log_gross_income', 'log_operating_expense', 'log_business_sales',
    'profit_margin', 'revenue_efficiency',
    'financial_stress_score', 'formality_score',
    'digital_access_score', 'insurance_gap', 'extends_credit',
    'survey_coverage',
    'cluster'
]
df_orange = df[orange_cols].copy()
cluster_map = {0: 'Large_Informal', 1: 'Stable_Formal',
               2: 'High_Risk_Informal', 3: 'Established_Veterans'}
df_orange['cluster'] = df_orange['cluster'].map(cluster_map)
df_orange.to_csv('data/msme_clustered_for_orange.csv', index=False)
print(f'Exported {len(df_orange)} rows to data/msme_clustered_for_orange.csv')
```

Install Orange if you haven't: download from [orangedatamining.com/download](https://orangedatamining.com/download/) or run `pip install orange3`.

---

### Step 1 — Load Data into Orange

1. Open Orange and create a new workflow (**File → New**).
2. From the left panel under **Data**, drag a **File** widget onto the canvas.
3. Double-click the File widget. Click **Browse** and select `data/msme_clustered_for_orange.csv`.
4. Orange will auto-detect column types. **You must manually set one thing:**
   - Find the **`cluster`** column in the column list → change its **Role** from "feature" to **"target"** (click the role dropdown next to it). This tells Orange that `cluster` is what we want the tree to predict.
   - Check that text columns (Nationality, Gender, has_insurance, etc.) are set to **Categorical** and numeric columns (log_gross_income, profit_margin, etc.) are set to **Numeric**. Orange usually gets this right automatically.
5. Click **Apply** or close the widget.

---

### Step 2 — Build a Multi-Class Decision Tree

This tree predicts which of the 4 segments a business belongs to. The features it uses to split are the risk drivers.

1. From the left panel under **Model**, drag a **Tree** widget onto the canvas.
2. **Connect them:** click the File widget's output port (right edge) and drag a line to the Tree widget's input port (left edge).
3. Double-click the Tree widget and set:
   - **Induce binary tree:** checked
   - **Limit the depth:** 5 (keeps the tree readable — deeper trees are harder to interpret)
   - **Do not split subsets smaller than:** 50
   - **Stop when majority reaches (%):** 95
4. From **Visualize**, drag a **Tree Viewer** widget onto the canvas.
5. Connect **Tree → Tree Viewer** (drag from Tree's output to Tree Viewer's input).
6. Double-click the Tree Viewer to see the full tree.

**What you will see:**
- A tree diagram with boxes (nodes) connected by lines (branches).
- The **root node at the top** shows the single most important feature — the one that best separates all four segments.
- Each node shows: the feature name, the split threshold (e.g., `financial_stress_score ≤ 2.5`), how many businesses fall into each branch, and the majority class.
- Leaf nodes at the bottom show the final predicted segment.

**Take a screenshot of this tree.**

---

### Step 3 — Evaluate the Tree

This tells you how accurate the tree is — whether the segments are genuinely distinct.

1. From **Evaluate**, drag **Test & Score** onto the canvas.
2. Connect **File → Test & Score** (data input, connect to the left "Data" port).
3. Connect **Tree → Test & Score** (learner input, connect to the top "Learner" port).
4. Double-click Test & Score. Select **10-Fold Cross Validation** (default). Click **Apply**.
5. Look at the results table:
   - **CA** (Classification Accuracy): what % of businesses the tree classifies correctly.
   - **F1**: a balanced accuracy measure (accounts for unequal cluster sizes).
   - **AUC**: how well the tree separates each segment from the others (0.5 = random, 1.0 = perfect).
6. From **Evaluate**, drag a **Confusion Matrix** widget. Connect **Test & Score → Confusion Matrix**.
7. Double-click the Confusion Matrix. It shows a grid: rows = actual segment, columns = predicted segment. Diagonal = correct predictions. Off-diagonal = mistakes. This tells you which segments the tree confuses with each other.

**Take screenshots of both the Test & Score results and the Confusion Matrix.**

---

### Step 4 — Per-Segment Trees (Top 5 Risk Drivers)

The multi-class tree shows the overall picture, but to find the **top 5 risk drivers specific to each segment**, you need four separate binary trees — each asking: *"What makes a business belong to THIS segment vs. all others?"*

**For each of the 4 segments, do the following:**

1. From **Data**, drag a **Python Script** widget onto the canvas.
2. Connect **File → Python Script**.
3. Double-click the Python Script widget and paste this code:

```python
import numpy as np
from Orange.data import Domain, DiscreteVariable, Table

# ── Change this name for each segment ──
target_segment = "High_Risk_Informal"
# Other options: "Large_Informal", "Stable_Formal", "Established_Veterans"

# Get the cluster column and create a binary target
cluster_vals = [str(d[in_data.domain.class_var]) for d in in_data]
binary = np.array([1 if v == target_segment else 0 for v in cluster_vals])

# Build new dataset with binary target
new_class = DiscreteVariable(f"is_{target_segment}", values=["No", "Yes"])
new_domain = Domain(in_data.domain.attributes, new_class)
out_data = in_data.transform(new_domain)
for i, val in enumerate(binary):
    out_data[i].set_class(val)
```

4. Click **Run** (play button at the bottom of the script editor).
5. Connect **Python Script → Tree → Tree Viewer** (drag a new Tree and Tree Viewer for each).
6. Configure each Tree widget the same way as Step 2 (depth 5, min 50, binary tree).
7. Open each Tree Viewer to see the tree.

**Reading the per-segment tree:**

The **root node** (first split) is **risk driver #1** for that segment. The second level of splits gives drivers #2 and #3. Continue down to find all 5. Record them in a table:

| Rank | Large Informal | Stable Formal | High Risk Informal | Established Veterans |
|---|---|---|---|---|
| 1 | *(root split)* | *(root split)* | *(root split)* | *(root split)* |
| 2 | *(2nd level)* | ... | ... | ... |
| 3 | *(2nd level)* | ... | ... | ... |
| 4 | *(3rd level)* | ... | ... | ... |
| 5 | *(3rd level)* | ... | ... | ... |

**Take a screenshot of each of the 4 Tree Viewers.**

---

### Step 5 — Feature Importance Ranking

This gives a complementary view — instead of tree splits, it ranks every feature by how much information it provides about the target.

1. From **Data**, drag a **Rank** widget onto the canvas.
2. Connect **File → Rank**.
3. Double-click the Rank widget. It shows a table with every feature ranked by:
   - **Information Gain:** how much knowing this feature reduces uncertainty about the cluster.
   - **Gain Ratio:** information gain adjusted for features with many categories.
   - **Gini:** how well the feature separates the clusters (lower Gini = better separation).
4. The top-ranked features here should largely agree with the tree's top splits. If a feature ranks highly in Rank but doesn't appear in the tree, it may be correlated with another feature that the tree chose instead.

**Take a screenshot of the Rank table.**

---

### How to Interpret and Write Up the Results

#### Translating Tree Splits into Business Insights

For each risk driver, write one sentence explaining **what it means for a lending decision**. Use the cluster profiles from Part 1 to add context. Examples based on likely outcomes:

**If `financial_stress_score` is the #1 driver for High Risk Informal:**
> "The strongest predictor of high-risk status is the number of concurrent financial distress signals — cash flow problems, reliance on informal lenders, liquidity constraints. A business showing 4+ of these 5 signals is almost certainly in the high-risk segment. For the lender, this means that extending credit without first addressing the underlying cash flow instability carries extreme default risk."

**If `formality_score` separates Stable Formal from others:**
> "Formal financial integration — having insurance, a bank account, a debit card, and tax compliance — is the clearest marker of a bankable MSME. Businesses with formality scores above 2.0 are overwhelmingly in the Stable Formal segment. These businesses have verifiable financial histories that a lender can assess using standard credit evaluation."

**If `customer_age` or `years_in_operation` drives Established Veterans:**
> "Business longevity (20+ years) combined with owner age (55+) defines this segment. The business has proven it can survive, but the owner is approaching retirement. The risk here is not default — it is succession. A lender should consider whether the business will outlast the current owner."

**If `profit_margin` separates Large Informal from High Risk Informal:**
> "Profitability is the line between moderate and extreme risk. Both segments are informal, but Large Informal businesses are making money (positive margin) while High Risk Informal businesses are losing money (margin of −2.1). A lender can use profit margin as a quick filter: informal + profitable = worth engaging; informal + loss-making = needs intervention first."

**If `Nationality` appears as a top driver:**
> "Country of operation is a strong predictor because each market has fundamentally different financial infrastructure. This is not an individual business characteristic — it reflects the operating environment. The lender should interpret this as: risk strategies must be tailored per market, not applied uniformly across all four countries."

---

### Recommended Workflow Layout in Orange (Main Clustering)

```
                         ┌──→ [Tree Viewer]        ← Screenshot: main tree
[File] ──→ [Tree] ──────┘
   │
   ├──→ [Test & Score] ──→ [Confusion Matrix]     ← Screenshot: accuracy + confusion
   │
   ├──→ [Rank]                                     ← Screenshot: feature rankings
   │
   ├──→ [Python Script: High_Risk_Informal] ──→ [Tree] ──→ [Tree Viewer]  ← Screenshot
   ├──→ [Python Script: Large_Informal] ──→ [Tree] ──→ [Tree Viewer]      ← Screenshot
   ├──→ [Python Script: Stable_Formal] ──→ [Tree] ──→ [Tree Viewer]       ← Screenshot
   └──→ [Python Script: Established_Veterans] ──→ [Tree] ──→ [Tree Viewer]← Screenshot
```

---

### Step 6 — Within-Country Root Cause Analysis

The main trees above explain what drives the 4 global segments — but since those segments are largely country-dominated, the trees may just learn "Nationality = Uganda → High Risk." The within-country clustering from Part 1 found sub-segments *within* each market. Building trees on these reveals **country-specific risk drivers** — much more actionable for a lender operating in a single market.

#### 6a — Export Within-Country Data

Add and run this cell at the end of the notebook (after the within-country clustering cells):

```python
# Export within-country clustered data for Orange
# One file per country, with sub-cluster labels as the target

wc_export_features = [
    'Gender', 'customer_age', 'years_in_operation',
    'has_insurance', 'active_loan_holder', 'has_internet_banking',
    'has_debit_card', 'credit_card_ownership', 'compliance_income_tax',
    'current_problem_cash_flow', 'liquidity_constraint',
    'uses_informal_lender', 'uses_friends_family_savings',
    'funding_access_barrier', 'mobile_wallet_access',
    'extends_customer_credit', 'inventory_shrinkage_risk',
    'financial_tracking',
    'log_gross_income', 'log_operating_expense', 'log_business_sales',
    'profit_margin', 'revenue_efficiency',
    'financial_stress_score', 'formality_score',
    'digital_access_score', 'insurance_gap', 'extends_credit',
]

# Sub-cluster names from Part 1 within-country analysis
subcluster_names = {
    'Ghana': {0: 'Young_Mainstream', 1: 'Established_Formal'},
    'Kenya': {0: 'Mature_Informal', 1: 'Young_Hustlers', 2: 'Formal_High_Earners'},
    'Uganda': {0: 'Severely_Distressed', 1: 'Veteran_Survivors',
               2: 'Lower_Stress_Informal', 3: 'Emerging_Formal'},
    'DRC': {0: 'Profitable_Mid_Formal', 1: 'Small_Struggling',
            2: 'Stressed_Profitable', 3: 'Highly_Formal_Elite',
            4: 'Veteran_Established'}
}

for country in ['Ghana', 'Kenya', 'Uganda', 'DRC']:
    mask = df['Nationality'] == country
    df_c = df.loc[mask, wc_export_features].copy()
    df_c['subcluster'] = df.loc[mask, 'within_cluster'].map(subcluster_names[country])
    fname = f'data/orange_{country.lower()}_subclusters.csv'
    df_c.to_csv(fname, index=False)
    print(f'{country}: {len(df_c)} rows, {df_c["subcluster"].nunique()} sub-clusters → {fname}')
```

This creates 4 files:
- `data/orange_ghana_subclusters.csv`
- `data/orange_kenya_subclusters.csv`
- `data/orange_uganda_subclusters.csv`
- `data/orange_drc_subclusters.csv`

#### 6b — Build Within-Country Trees in Orange

You can do this in the same Orange workflow or a new one. **Repeat for each country:**

1. Drag a new **File** widget. Load the country's CSV (e.g., `orange_uganda_subclusters.csv`).
2. Set `subcluster` as the **target** column (same as Step 1).
3. Connect **File → Tree → Tree Viewer** (same settings: depth 5, min 50, binary tree).
4. Connect **File → Rank** to see feature importance for that country.
5. Open the Tree Viewer — the tree now shows what separates sub-clusters *within that country*.

**What to look for in each country's tree:**

**Ghana (2 sub-clusters: Young Mainstream vs. Established Formal):**
- The tree should split primarily on `years_in_operation` and `customer_age` — the two groups differ mainly on experience. Look for whether `formality_score` also appears, since Established Formal has 4x higher formality.
- *Lending insight:* Longer tenure → more formal → lower risk, even though margins are thinner.

**Kenya (3 sub-clusters: Mature Informal, Young Hustlers, Formal High-Earners):**
- Expect `formality_score` to be the #1 split — it's the only feature that sharply separates Formal High-Earners (1.49) from the other two (~0.1).
- A second split on `years_in_operation` or `customer_age` should separate Young Hustlers from Mature Informal.
- *Lending insight:* The 59% majority (Young Hustlers) are profitable but invisible to formal lenders. The tree can reveal which secondary features (digital access? financial tracking?) might predict formalisation readiness.

**Uganda (4 sub-clusters: Severely Distressed, Veteran Survivors, Lower-Stress Informal, Emerging Formal):**
- This is the most important tree. The main clustering labelled all Uganda as "high risk" — this tree shows the internal variation.
- Expect `financial_stress_score` to be the root split — it ranges from 2.49 (Lower-Stress) to 5.00 (Severely Distressed).
- `formality_score` should separate Emerging Formal (2.01) from everyone else (~0.0).
- `years_in_operation` should isolate Veteran Survivors (20+ years).
- *Lending insight:* The 21% Lower-Stress Informal sub-cluster is the realistic entry point for micro-credit in Uganda. The tree tells you exactly which features flag a Ugandan business as "less distressed than average."

**DRC (5 sub-clusters: Profitable Mid-Formal, Small Struggling, Stressed Profitable, Highly Formal Elite, Veteran Established):**
- The richest tree. Expect `formality_score` and `profit_margin` as top splits.
- `financial_stress_score` should separate Stressed Profitable (2.87) from Profitable Mid-Formal (1.25) — these two have similar income and formality but very different stress.
- `years_in_operation` should isolate Veteran Established (24 years).
- *Lending insight:* The distinction between "profitable + stressed" vs. "profitable + calm" is critical. Stressed Profitable businesses may be growing fast and need working capital; Profitable Mid-Formal businesses are stable and can handle standard term loans.

#### 6c — Record Within-Country Risk Drivers

Fill in a table for each country. Example for Uganda:

| Rank | Risk Driver | Split Value | What It Means |
|---|---|---|---|
| 1 | `financial_stress_score` | ≤ 2.5 | Separates the severely distressed majority from the recoverable minority |
| 2 | `formality_score` | > 1.0 | Isolates the 7% with formal financial access — best lending candidates |
| 3 | `years_in_operation` | > 15 | Identifies long-surviving businesses that are stressed but resilient |
| 4 | *(from tree)* | ... | ... |
| 5 | *(from tree)* | ... | ... |

**Take a screenshot of each country's Tree Viewer and Rank table.**

---

### Workflow Layout — Within-Country Trees

```
[File: Ghana]  ──→ [Tree] ──→ [Tree Viewer]    ← Screenshot
               ──→ [Rank]                       ← Screenshot

[File: Kenya]  ──→ [Tree] ──→ [Tree Viewer]    ← Screenshot
               ──→ [Rank]                       ← Screenshot

[File: Uganda] ──→ [Tree] ──→ [Tree Viewer]    ← Screenshot
               ──→ [Rank]                       ← Screenshot

[File: DRC]    ──→ [Tree] ──→ [Tree Viewer]    ← Screenshot
               ──→ [Rank]                       ← Screenshot
```

---

### Total Screenshots for Part 2

| What | Count |
|---|---|
| Main multi-class tree | 1 |
| Test & Score results | 1 |
| Confusion Matrix | 1 |
| Rank table (all segments) | 1 |
| Per-segment binary trees (4 segments) | 4 |
| Within-country trees (4 countries) | 4 |
| Within-country rank tables (4 countries) | 4 |
| **Total** | **16** |

All screenshots go into the PDF report submission alongside the notebook.

---

## Part 3: Risk Strategy Recommendations (Pending)

Segment-specific risk mitigation strategies based on Parts 1 and 2.
