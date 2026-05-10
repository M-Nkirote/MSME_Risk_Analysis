# Risk Segmentation & Diagnostic Strategy for MSME Lending in Africa

**Dataset:** 9,618 MSMEs across Kenya, Uganda, DRC, and Ghana

---

## 1. Executive Summary

This report presents a data-driven risk segmentation of 9,618 Micro, Small, and Medium Enterprises (MSMEs) across four African markets to inform the institution's lending strategy as it expands into these regions.

Using unsupervised clustering (K-Means) on financial performance, stress indicators, and formality metrics, we identified **four distinct risk segments**: Stable Formal (22%), Large Informal (42%), Established Veterans (11%), and High Risk Informal (25%). Decision tree classifiers built in Orange Data Mining achieved **94.5% classification accuracy**, confirming these segments represent genuinely distinct risk populations.

**Three critical findings for the Board:**

1. **Financial access - not income - is the primary risk separator.** The top predictive features are internet banking access, debit card ownership, and active loan status. Businesses with even minimal formal financial integration are fundamentally different from those without. The institution's first screening question should be about financial access, not revenue.

2. **Country-level strategies are essential.** Three of four segments are dominated by a single nationality, reflecting structural differences in financial ecosystems. A uniform credit policy across all four markets would be poorly calibrated. Within-country analysis reveals further sub-segmentation: Uganda alone contains four distinct risk sub-groups.

3. **The highest-risk segment is predominantly female-owned.** The High Risk Informal cluster (25% of dataset, 98% Ugandan) is 67% women-owned. Any lending intervention in this segment is, in practice, a women's financial inclusion programme. However, within-country analysis shows 21% of Ugandan businesses are meaningfully less distressed than the majority - these represent the realistic entry point for micro-credit.

---

## 2. Problem Statement

The institution seeks to expand lending operations into Kenya, Uganda, DRC, and Ghana. These markets contain thousands of MSMEs with diverse risk profiles, operating environments, and levels of financial integration. Without a structured understanding of which businesses are creditworthy, which need pre-credit support, and which risk factors dominate each market, the institution faces two dangers: rejecting viable borrowers due to blanket risk aversion, or incurring high default rates by lending indiscriminately.

The dataset contains 9,618 business records spanning business characteristics, owner demographics, financial behaviour, and current challenges. Our task is to segment these businesses into actionable risk groups, identify what drives each group's risk profile, and recommend tailored lending strategies.

---

## 3. Methodology

The analysis followed three phases:

**Phase 1 - Customer Segmentation (Unsupervised Learning)**

- Data cleaning: type standardisation, country-aware missing value imputation (4-step strategy addressing structural survey differences between countries), IQR-based outlier winsorisation, and log transformation of financial variables.
- Feature engineering: 9 domain-relevant features including profit margin, financial stress score (0-5), formality score (0-5), revenue efficiency, digital access score, and insurance gap.
- Clustering: K-Means (k=4) validated with elbow method, silhouette scores (0.29), and agglomerative clustering comparison (ARI = 0.45). A sensitivity check confirmed results are robust to inclusion/exclusion of survey coverage as a feature (ARI = 0.64).
- Within-country clustering: K-Means run separately for each country (Ghana k=2, Kenya k=3, Uganda k=4, DRC k=5) to identify sub-segments independent of nationality.

**Phase 2 - Root Cause Analysis (Orange Data Mining)**

- Multi-class decision tree (depth 5, min 50 leaf instances) with 10-fold cross-validation: CA = 94.5%, AUC = 0.972.
- Four one-vs-rest binary decision trees to isolate per-segment risk drivers.
- Feature importance ranking via information gain, gain ratio, and Gini index.
- Within-country decision trees for all four markets.

**Phase 3 - Risk Strategy Recommendations**

Segment-specific and country-specific lending strategies grounded in the decision tree outputs, with each recommendation linked to a specific risk driver.

---

## 4. Findings

### 4.1 Four Risk Segments

| Tier | Segment | Size | Key Profile | Risk Level |
|---|---|---|---|---|
| 1 | **Stable Formal** | 22% | Best margins (0.45), formality 2.03, DRC-dominated (74%) | Lowest |
| 2 | **Large Informal** | 42% | Largest income, formality 0.28, Ghana (46%) + Kenya (52%) | Moderate |
| 3 | **Established Veterans** | 11% | 22-yr tenure, owners avg. 59 yrs, cross-country, slightly loss-making | Moderate-High |
| 4 | **High Risk Informal** | 25% | Margin -2.13, stress 4.39, formality 0.08, 98% Uganda, 67% female | Highest |

### 4.2 Top Risk Drivers (Decision Tree Analysis)

The overall feature ranking by information gain reveals that **financial access indicators** dominate over financial performance:

| Rank | Feature | Info. Gain | Significance |
|---|---|---|---|
| 1 | Internet banking access | 1.062 | Strongest single separator across all segments |
| 2 | Nationality | 1.030 | Country structure drives risk more than individual characteristics |
| 3 | Survey coverage | 1.003 | Data completeness varies systematically by country |
| 4 | Debit card ownership | 1.002 | Verifiable transaction history reduces information asymmetry |
| 5 | Active loan status | 0.957 | Existing borrowers cluster into Stable Formal; unsurveyed into High Risk |

Per-segment binary trees identified the root drivers: `active_loan_holder` = "Not Surveyed" predicts High Risk Informal at 96.4%; `formality_score` ≤ 1 excludes businesses from Stable Formal at 99.3%; `has_internet_banking` = "Don't have now" predicts Large Informal at 96.7%; `years_in_operation` > 12 identifies Established Veterans.

### 4.3 Within-Country Risk Drivers

Each country has distinct internal dynamics:

- **Uganda:** Financial stress score is the dominant driver (info. gain 0.667). A threshold of stress ≤ 2.5 cleanly identifies 21% of Ugandan businesses as "Lower-Stress Informal" with 99.8% accuracy - these are the realistic micro-credit candidates.
- **Kenya:** Formality score dominates (info. gain 0.519). 20% of Kenyan businesses qualify as "Formal High-Earners" based on debit card and tax compliance - immediately bankable.
- **DRC:** Five sub-segments driven by stress, income, formality, and profit margin. The distinction between "Stressed Profitable" (growing, needs working capital) and "Profitable Mid-Formal" (stable, suits term loans) is critical.
- **Ghana:** Tenure is the primary axis. The 7-year mark separates mainstream young businesses from established formal ones.

---

## 5. Recommendations

### 5.1 Segment-Specific Lending Strategies

**Stable Formal (22%) - Retention-Focused:**
- Pre-approved revolving credit using existing banking records for automated underwriting.
- Tiered loyalty pricing (rate reductions per on-time cycle) to prevent churn to competitors.
- Cross-sell insurance, savings, and digital payment products to deepen relationships.

**Large Informal (42%) - Formalisation Pipeline:**
- Alternative data underwriting via mobile money partnerships (M-Pesa, MTN Mobile Money).
- Assisted digital onboarding as part of loan applications - creates verifiable data trails.
- Incentivised formalisation: interest rates reduce as the business achieves milestones (tax registration, bank account, insurance).

**Established Veterans (11%) - Succession-Aware:**
- Succession-linked financing: loans beyond 36 months require a designated secondary operator.
- Shorter loan tenors (12-18 months) with renewal, reassessing viability at each cycle.
- Digital transition support for technologically stagnant businesses.

**High Risk Informal (25%) - Graduated Entry:**
- Tranche disbursement: minimal initial loans ($50-100), higher limits only after 3-5 on-time cycles.
- Pre-credit financial literacy targeting the 21% Lower-Stress sub-segment in Uganda before any lending.
- Women-focused product design (67% female): flexible repayment aligned with market-day cash flows.
- Alternative credit history building from mobile money, utility payments, and savings group participation.

### 5.2 Market-Level Priorities

| Market | Entry Strategy | Priority Sub-segment | First Product |
|---|---|---|---|
| **DRC** | Broadest opportunity - 5 sub-segments, highest formality | Highly Formal Elite (9%) | Pre-approved term loans |
| **Kenya** | Three-tier market - target the formal tier first | Formal High-Earners (20%) | Digital revolving credit |
| **Ghana** | Tenure-driven - use experience as pre-screening | Established Formal (32%) | Standard business loans |
| **Uganda** | Highest risk - graduated entry only | Lower-Stress Informal (21%) | Micro-stepping tranche loans |

### 5.3 Institutional Recommendations

1. **Adopt formality score > 1 as the primary credit gateway.** The decision tree confirms this single threshold excludes 99.3% of the highest-risk businesses from standard products.
2. **Invest in mobile money data partnerships** before entering Kenya and Ghana. The largest segment (42%) is profitable but formally invisible - the institution that solves the verification problem captures this market.
3. **Do not treat Uganda as uniformly high-risk.** Within-country analysis identifies 28% of Ugandan businesses (Lower-Stress + Emerging Formal) as meaningfully different from the distressed majority.
4. **Track financial stress score as an early warning indicator.** A stress score crossing 2.5 (Uganda) or 1.25 (DRC) signals segment migration - the borrower's risk profile is deteriorating.
