# Application Credit Risk Scorecard

## Project Overview

This project develops an end-to-end **retail application credit risk scorecard** using the Kaggle **Give Me Some Credit** dataset. The objective is to build an interpretable, production-ready model that:

- **Predicts serious delinquency risk** within 24 months (90+ DPD threshold)
- **Supports approval/reject decisions** at the application stage
- **Enables risk segmentation** for portfolio management
- **Transforms model output** into a business-friendly credit score for easy interpretation

The final model is based on **WOE-transformed predictors** and a **logistic regression classifier**, which is a standard and highly interpretable modelling approach for credit scorecards.

---

## Business Context

### Model Objective
- **Use Case**: Approval/Reject decisions at application stage
- **Risk Segmentation**: Support risk-based decision making
- **Future Applications**: Potential pricing and limit assignment
- **Delinquency Threshold**: 90+ Days Past Due (DPD)
- **Performance Window**: 24 months

### Target Population
- **Customer Type**: Retail customers (individuals only)
- **Geographic Scope**: Applicants included in Kaggle "Give Me Some Credit" competition dataset

---

## Dataset

Dataset: **Give Me Some Credit**  
Source: Kaggle competition dataset

The target variable is:

| Variable | Meaning |
|---|---|
| `SeriousDlqin2yrs` | Whether the applicant experienced serious delinquency (90+ DPD) within two years |

Target encoding:

| Value | Interpretation |
|---:|---|
| 0 | Good customer (no serious delinquency) |
| 1 | Bad customer (serious delinquency within 24 months) |

A "Bad" customer is defined as one who experiences 90+ days past due within the two-year performance window.

---

## Project Structure

```text
Application_Creditrisk_Scorecard/
│
├── data/
│   ├── raw/
│   │   ├── cs-training.csv
│   │   ├── cs-test.csv
│   │   ├── Data Dictionary.xls
│   │   └── sampleEntry.csv
│   │
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   ├── train_woe.csv
│   │   ├── validation_woe.csv
│   │   ├── test_woe.csv
│   │   ├── train_scored.csv
│   │   ├── validation_scored.csv
│   │   └── test_scored.csv
│   │
│   └── outputs/
│       ├── binning_results.xlsx
│       ├── data_dictionary.csv
│       ├── score_distribution.csv
│       ├── score_decile_analysis.csv
│       └── cutoff_fit_summary.csv
│
├── notebooks/
│   ├── 0_business_understanding.ipynb
│   ├── 0_data_extraction.ipynb
│   ├── 01_data_collection_and_quality.ipynb
│   ├── 02_sampling.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_outlier_analysis.ipynb
│   ├── 05_variable_screening.ipynb
│   ├── 06_binning_methodology.ipynb
│   ├── 07_woe_transformation.ipynb
│   ├── 08_feature_selection.ipynb
│   ├── 09_model_performance.ipynb
│   ├── 10_scorecard_scaling.ipynb.ipynb
│   ├── 11_cutoff_determination.ipynb
│   ├── 12_score_run_and_fit.ipynb
│   └── 13.validation_and_monitoring.ipynb
│
├── requirements.txt
├── .gitignore
└── .env
```

---

## Main Modelling Objective

The model estimates the **probability of default / serious delinquency**:

\[
PD = P(Y = 1 \mid X)
\]

Where:

| Symbol | Description |
|----------|-------------|
| \(PD\) | Probability of Default |
| \(Y = 1\) | Customer is classified as **Bad** (default / serious delinquency) |
| \(Y = 0\) | Customer is classified as **Good** |
| \(X\) | Set of applicant characteristics used by the model |

The model can support:

- Approval / reject decisions
- Risk segmentation
- Cut-off determination
- Manual review strategy
- Score-based portfolio monitoring

---

## Getting Started

### Prerequisites & Dependencies

**System Requirements:**
- Python 3.8+
- pip or conda package manager
- Git (for version control)

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Required Libraries:**
- **Data Processing**: pandas, numpy
- **Modeling**: scikit-learn, statsmodels
- **Binning**: optbinning (optimal binning for credit scoring)
- **Data I/O**: openpyxl, xlrd (Excel support)
- **Visualization**: matplotlib
- **Notebooks**: jupyter
- **Data Source**: kagglehub (Kaggle API wrapper)

---

### Environment Configuration

**Setup API Access:**

1. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```

2. Obtain your Kaggle API token:
   - Go to https://www.kaggle.com/account
   - Click "Create New Token" (downloads `kaggle.json`)
   - Extract the token value

3. Add to `.env`:
   ```
   KAGGLE_API_TOKEN=your_kaggle_token_here
   ```

4. The notebooks will load this token automatically

**Note**: The `.env` file is listed in `.gitignore` and should never be committed to version control.

---

## 0. Business Understanding

**Notebook**: `0_business_understanding.ipynb`

This notebook establishes the business context and objectives for the credit scorecard project.

### Key Definitions

**Model Objective**: Approval/Reject decisions at application stage

**Target Definition**:
- Good = 0 (no serious delinquency)
- Bad = 1 (90+ DPD within 24 months)
- Performance Window: 24 months

**Population Definition**:
- Retail customers
- Individuals only

### Business Use Cases

- Approval / reject decisions
- Risk segmentation
- Cut-off determination
- Potential pricing and limit assignment

---

## 0. Data Extraction

**Notebook**: `0_data_extraction.ipynb`

This notebook handles downloading the Kaggle dataset and preparing raw data files.

### Process

```
Kaggle API Token (from .env)
    ↓
Download Competition Dataset via kagglehub
    ↓
Extract Files (cs-training.csv, cs-test.csv, etc.)
    ↓
Copy to data/raw/
```

### Output Files Created

- `data/raw/cs-training.csv` - Full training dataset (~150K records)
- `data/raw/cs-test.csv` - Test dataset (~101K records)
- `data/raw/Data Dictionary.xls` - Feature definitions and metadata
- `data/raw/sampleEntry.csv` - Sample submission format

---

## 1. Data Collection & Quality Assessment

**Notebook**: `01_data_collection_and_quality.ipynb`

This notebook performs comprehensive data quality checks and profiling of the raw dataset. It validates data integrity, identifies missing values, and provides descriptive statistics for modeling.

### Dataset Summary

**Size & Composition:**
- **Records**: 150,000 applicants
- **Variables**: 11 (1 target + 10 predictors)
- **Data Type**: All numeric

**Target Variable Distribution:**

| Class | Count | Percentage |
|-------|-------|-----------|
| Good (0) | 140,024 | 93.35% |
| Bad (1) | 9,976 | 6.65% |

**Key Insight**: Highly imbalanced dataset with ~14:1 ratio (Good:Bad). This imbalance requires careful modeling strategies.

### Data Quality Findings

**Missing Values:**

| Variable | Missing Count | Missing % | 
|----------|---------------|-----------|
| MonthlyIncome | 29,731 | 19.82% |
| NumberOfDependents | 3,924 | 2.62% |
| All Others | 0 | 0.00% |

**Data Issues Identified:**
- **Extreme Values**: RevolvingUtilizationOfUnsecuredLines has max of 50,708 (should be ratio 0-1) → Outliers present
- **Suspicious Age**: Minimum age of 0 years → Data quality issue
- **High Debt Ratios**: Max DebtRatio of 329,664 (likely data errors or extreme cases)
- **Income Spikes**: MonthlyIncome max of 3,008,750 with many zero values

### Feature Overview

**Variable Categories:**

| Category | Variables | Count |
|----------|-----------|-------|
| Demographic | `age` | 1 |
| Delinquency Behavior | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate` | 3 |
| Credit Portfolio | `RevolvingUtilizationOfUnsecuredLines`, `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines` | 3 |
| Financial | `DebtRatio`, `MonthlyIncome`, `NumberOfDependents` | 3 |

### Summary Statistics

**Key Findings:**
- **Age**: Mean 52.3 years (SD 14.8), range 0-109 years
- **DebtRatio**: Mean 353.0 (SD 2,037.8) → High variability indicating outliers
- **MonthlyIncome**: Mean $6,670 (SD $14,384.7) → Extreme values distorting mean
- **Credit Behavior**: Most applicants have 0 delinquencies (median = 0 for all delinquency variables)

---

## 2. Sampling Strategy

**Notebook**: `02_sampling.ipynb`

The dataset was split into three samples using stratified random sampling to preserve the target distribution (bad rate) across all samples.

### Portfolio Bad Rate

**Overall Portfolio Bad Rate**: 6.68%

| Class | Distribution |
|-------|--------------|
| Good (0) | 93.32% |
| Bad (1) | 6.68% |

### Sample Split

**Stratified sampling method**: Applied stratification on `SeriousDlqin2yrs` to ensure consistent bad rate across all samples.

| Sample | Records | Percentage | Bad Rate |
|--------|---------|-----------|----------|
| Train | 90,000 | 60.0% | 6.6844% |
| Validation | 30,000 | 20.0% | 6.6833% |
| Test | 30,000 | 20.0% | 6.6833% |

### Key Achievement

**Perfect Stratification**: The bad rate across all three samples is virtually identical (6.6833-6.6844%), confirming that stratified sampling successfully preserved the portfolio bad rate distribution. This is critical for:

- **Fair Model Evaluation**: Performance metrics are comparable across samples
- **Unbiased Validation**: The validation and test samples have the same risk profile as the training set
- **Reliable Generalization**: Model performance on test set accurately reflects real-world deployment performance

### Output Files

The stratified samples are saved as:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

These files are used in all subsequent modeling notebooks (EDA, binning, WOE transformation, model development, etc.).

---

## 3. Exploratory Data Analysis

**Notebook**: `03_exploratory_data_analysis.ipynb`

Comprehensive exploratory analysis of the training sample across demographic, credit behavior, and financial dimensions.

### Portfolio Overview

**Training Sample Size**: 90,000 observations  
**Bad Rate**: 6.68% | **Good Rate**: 93.32%

### Variable Analysis & Key Statistics

**Descriptive Statistics by Variable:**

| Variable | Count | Mean | Std Dev | Min | P25 | Median | P75 | P95 | P99 | Max |
|----------|-------|------|---------|-----|-----|--------|-----|-----|-----|-----|
| age | 90,000 | 52.31 | 14.76 | 0 | 41 | 52 | 63 | 78 | 87 | 109 |
| RevolvingUtilizationOfUnsecuredLines | 90,000 | 6.61 | 267.96 | 0 | 0.03 | 0.15 | 0.56 | 1.00 | 1.10 | 50,708 |
| DebtRatio | 90,000 | 356.95 | 2,336.63 | 0 | 0.18 | 0.37 | 0.86 | 2,433 | 4,948 | 329,664 |
| MonthlyIncome | 72,204* | 6,616 | 13,387 | 0 | 3,400 | 5,400 | 8,248 | 14,600 | 25,000 | 3,008,750 |
| NumberOfOpenCreditLinesAndLoans | 90,000 | 8.48 | 5.18 | 0 | 5 | 8 | 11 | 18 | 25 | 58 |
| NumberOfTimes90DaysLate | 90,000 | 0.27 | 4.27 | 0 | 0 | 0 | 0 | 1 | 3 | 98 |
| NumberOfTime30-59DaysPastDueNotWorse | 90,000 | 0.43 | 4.29 | 0 | 0 | 0 | 0 | 2 | 4 | 98 |
| NumberOfTime60-89DaysPastDueNotWorse | 90,000 | 0.25 | 4.25 | 0 | 0 | 0 | 0 | 1 | 2 | 98 |
| NumberRealEstateLoansOrLines | 90,000 | 1.02 | 1.14 | 0 | 0 | 1 | 2 | 3 | 5 | 54 |
| NumberOfDependents | 87,691* | 0.75 | 1.11 | 0 | 0 | 0 | 1 | 3 | 4 | 20 |

*Count reflects non-missing values

### Missing Value Analysis

| Variable | Missing Count | Missing % | Action |
|----------|--------------|-----------|--------|
| MonthlyIncome | 17,796 | 19.77% | **High** - Requires imputation |
| NumberOfDependents | 2,309 | 2.57% | **Low** - Can use mean/mode imputation |
| All Others | 0 | 0.00% | Complete |

### Key Observations by Variable

**Delinquency Variables:**
- Most applicants have zero delinquencies (median = 0)
- Small proportion have multiple delinquencies (up to 98 times)
- Right-skewed distributions with extreme outliers

**Financial Variables:**
- **DebtRatio**: Extreme range (max 329,664 vs median 0.37) indicating severe outliers
- **MonthlyIncome**: Max 3,008,750 vs median 5,400 (extreme outliers present)
- **RevolvingUtilizationOfUnsecuredLines**: Theoretical max should be ~1.0 but observed max is 50,708

**Demographic:**
- **Age**: Reasonable range 0-109 years (age=0 appears to be data quality issue)
- Mean age 52.3 years with SD 14.76 (moderate variation)

### Correlation Analysis

**High Correlations Detected** (Pearson correlation > 0.98):

| Variable 1 | Variable 2 | Correlation |
|------------|------------|------------|
| NumberOfTimes90DaysLate | NumberOfTime60-89DaysPastDueNotWorse | 0.9931 |
| NumberOfTime30-59DaysPastDueNotWorse | NumberOfTime60-89DaysPastDueNotWorse | 0.9875 |
| NumberOfTime30-59DaysPastDueNotWorse | NumberOfTimes90DaysLate | 0.9843 |

**Multicollinearity Risk**: The three delinquency variables are nearly perfectly correlated, indicating severe multicollinearity. This redundancy needs to be addressed through variable selection.

**Correlation with Target (SeriousDlqin2yrs):**

| Variable | Correlation with Target |
|----------|------------------------|
| NumberOfTimes90DaysLate | 0.1203 |
| NumberOfTime30-59DaysPastDueNotWorse | 0.1278 |
| NumberOfTime60-89DaysPastDueNotWorse | 0.1055 |
| age | -0.1127 |
| NumberOfDependents | 0.0444 |
| NumberOfOpenCreditLinesAndLoans | -0.0326 |
| MonthlyIncome | -0.0214 |

### Key Insights

1. **Imbalance Confirmed**: 6.68% bad rate creates significant class imbalance
2. **Missing Data**: 19.77% MonthlyIncome missing requires careful imputation strategy
3. **Outliers Present**: Multiple variables show extreme values far beyond reasonable ranges
4. **Multicollinearity**: Three delinquency variables are nearly perfectly correlated (need variable selection)
5. **Target Correlation**: 
   - Delinquency variables show strongest correlation with target (0.10-0.13)
   - Age shows negative correlation (-0.11) → older applicants have lower delinquency risk
   - Financial variables show weak correlation with target
6. **Data Quality**: Age range 0-109 suggests data cleaning needed before modeling

### Next Steps

1. **Outlier Treatment** (Notebook 04): Handle extreme values in DebtRatio, MonthlyIncome, RevolvingUtilization
2. **Variable Selection** (Notebook 05): Address multicollinearity among delinquency variables
3. **Binning & WOE** (Notebooks 06-07): Transform variables into predictive features

---

## 4. Outlier Analysis

**Notebook**: `04_outlier_analysis.ipynb`

Comprehensive outlier detection and analysis using percentile-based and IQR methods. Assessment of whether outliers represent data errors or valid extreme values requiring special handling.

### Percentile-Based Outlier Analysis

**Extreme Values Detected (Max/P99 Ratio):**

| Variable | P99 | Max | Max/P99 Ratio | Severity |
|----------|-----|-----|---------------|----------|
| RevolvingUtilizationOfUnsecuredLines | 1.10 | 50,708 | **46,305** | **Critical** |
| MonthlyIncome | 25,000 | 3,008,750 | **120** | **Critical** |
| DebtRatio | 4,948 | 329,664 | **67** | **Critical** |
| NumberOfTime60-89DaysPastDueNotWorse | 2.0 | 98 | 49 | **High** |
| NumberOfTimes90DaysLate | 3.0 | 98 | 33 | **High** |
| NumberOfTime30-59DaysPastDueNotWorse | 4.0 | 98 | 25 | **High** |
| NumberRealEstateLoansOrLines | 5.0 | 54 | 11 | **Moderate** |
| NumberOfDependents | 4.0 | 20 | 5 | Low |
| NumberOfOpenCreditLinesAndLoans | 25.0 | 58 | 2.3 | Low |
| age | 87.0 | 109 | 1.3 | Low |

### IQR Method Outlier Count

**Number of outliers detected using IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR):**

| Variable | Outlier Count |
|----------|----------------|
| DebtRatio | **18,712** (20.8%) |
| NumberOfTime30-59DaysPastDueNotWorse | **14,407** (16.0%) |
| NumberOfDependents | **7,896** (8.8%) |
| NumberOfTimes90DaysLate | **5,036** (5.6%) |
| NumberOfTime60-89DaysPastDueNotWorse | **4,633** (5.1%) |
| NumberOfOpenCreditLinesAndLoans | **2,459** (2.7%) |
| MonthlyIncome | **2,945** (3.3%) |
| RevolvingUtilizationOfUnsecuredLines | **464** (0.5%) |
| NumberRealEstateLoansOrLines | **476** (0.5%) |
| age | **27** (0.03%) |

### Data Quality Issues

**Age = 0 Records**: **195 records** (0.22% of training data)
- Represents a data quality issue and will be treated as missing during preprocessing

**Delinquency Variables - Special Value 98**: 
- Multiple delinquency variables exhibit a repeated maximum value of 98 (far below P99 range of 2-4)
- Likely represents a special coded value rather than actual delinquency counts
- Will be investigated during binning stage

### Business Validation

**Variable-Specific Insights:**

| Variable | Assessment |
|----------|-----------|
| **RevolvingUtilizationOfUnsecuredLines** | Values >100% may occur due to timing differences, reporting inconsistencies, or special credit arrangements. Legitimate business scenarios. |
| **MonthlyIncome** | Extremely high values (up to $3M) may represent genuine high-income applicants rather than data errors. |
| **DebtRatio** | Large ratios occur when customers have significant debt relative to very low reported income. Represents valid risk indicators. |
| **Delinquency Variables** | Value of 98 is likely a code rather than actual count. Will need special handling in binning. |
| **Age** | Age = 0 is clearly a data quality issue requiring treatment (missing imputation). |

### Treatment Decision

**No pre-binning outlier removal or transformation will be applied.**

**Rationale:**
1. **Methodology**: The scorecard uses **Optimal Binning + WoE transformation**, which naturally handles outliers through binning
2. **Information Preservation**: Extreme values may contain predictive signal and should not be discarded
3. **Robustness**: OptBinning is designed to create stable bins even with extreme values
4. **Transparency**: Original values retained for business validation

**Outlier Handling Strategy:**
- Original variable values are **retained unchanged**
- Extreme values will be **grouped by OptBinning** into meaningful bins
- Special coded values (age=0, delinquency=98) will be **handled during binning**
- Missing values will be **treated during imputation step**

### Key Insights

1. **Financial variables show most extreme outliers** (RevolvingUtilization, MonthlyIncome, DebtRatio)
2. **Delinquency variables have special coded value (98)** requiring investigation
3. **IQR method identifies many "outliers"** but many are legitimate extreme values with business meaning
4. **No extreme values are data errors** requiring removal - all represent valid business scenarios
5. **Age range is reasonable** (0-109) except for age=0 data quality cases

### Next Steps

1. **Variable Screening** (Notebook 05): Assess predictive power and eliminate redundant variables
2. **Optimal Binning** (Notebook 06): Create bins that handle outliers while preserving information
3. **WoE Transformation** (Notebook 07): Transform binned variables into predictive features

---

## 5. Variable Screening

**Notebook**: `05_variable_screening.ipynb`

Variable screening identifies predictive variables through Information Value (IV) analysis. IV quantifies how well each variable separates good and bad customers. Multicollinearity analysis informs later feature selection decisions.

### Missingness Analysis

**Missing Value Summary (Training Dataset):**

| Variable | Missing Count | Missing % |
|----------|---------------|-----------| 
| MonthlyIncome | 17,796 | **19.77%** |
| NumberOfDependents | 2,309 | **2.57%** |
| All other variables | 0 | 0.00% |

**Treatment Plan:**
- **MonthlyIncome**: Significant missing proportion (20%) will require imputation strategy during preprocessing
- **NumberOfDependents**: Minimal missing (2.57%) can be handled through mean/median imputation or category-based approach
- **Other variables**: Complete data across all 10 predictors

### Information Value Analysis

**IV Interpretation (Standard Credit Scoring Thresholds):**

### Weight of Evidence (WOE)

\[
WOE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right)
\]

WOE measures the separation between Good and Bad customers within a given bin.

### Information Value (IV)

\[
IV = \sum_{i} \left(\%Good_i - \%Bad_i\right) \cdot WOE_i
\]

IV measures the overall predictive strength of a variable by aggregating the discriminatory power of all bins.

| IV Range | Interpretation |
|---:|---|
| < 0.02 | Useless |
| 0.02 - 0.10 | Weak |
| 0.10 - 0.30 | Medium |
| 0.30 - 0.50 | Strong |
| ≥ 0.50 | Very Strong |

**Predictive Power Rankings:**

| Rank | Variable | IV | Strength | Classification |
|------|----------|----|----|---|
| 1 | RevolvingUtilizationOfUnsecuredLines | 2.22 | **Very Strong** | **Tier 1: Excellent** |
| 2 | NumberOfTimes90DaysLate | 1.70 | **Very Strong** | **Tier 1: Excellent** |
| 3 | NumberOfTime30-59DaysPastDueNotWorse | 1.46 | **Very Strong** | **Tier 1: Excellent** |
| 4 | NumberOfTime60-89DaysPastDueNotWorse | 1.16 | **Very Strong** | **Tier 1: Excellent** |
| 5 | age | 0.51 | **Very Strong** | **Tier 1: Excellent** |
| 6 | NumberOfOpenCreditLinesAndLoans | 0.17 | **Medium** | **Tier 2: Moderate** |
| 7 | DebtRatio | 0.16 | **Medium** | **Tier 2: Moderate** |
| 8 | MonthlyIncome | 0.16 | **Medium** | **Tier 2: Moderate** |
| 9 | NumberRealEstateLoansOrLines | 0.13 | **Medium** | **Tier 2: Moderate** |
| 10 | NumberOfDependents | 0.07 | **Weak** | **Tier 3: Limited** |

### Key Predictive Insights

**Tier 1 - Excellent Predictors (IV ≥ 0.50):**
- **Delinquency Variables Dominate**: The three past-due variables (90-day, 30-59 day, 60-89 day) are the strongest predictors of serious delinquency
- **Utilization Risk**: RevolvingUtilizationOfUnsecuredLines (IV=2.22) is the **single strongest predictor**
- **Demographic Factor**: Age (IV=0.51) is the only demographic variable with Very Strong predictive power
- **Combined Strength**: These 5 variables explain substantial discriminatory power between good and bad applicants

**Tier 2 - Moderate Predictors (0.10 < IV < 0.30):**
- NumberOfOpenCreditLinesAndLoans, DebtRatio, MonthlyIncome, NumberRealEstateLoansOrLines
- Contributing supplementary predictive information
- Will be included in binning and further evaluation

**Tier 3 - Limited Predictor (IV < 0.10):**
- NumberOfDependents (IV=0.07): Weak but retained for business relevance validation

### Multicollinearity Assessment

**High Correlation Among Delinquency Variables:**

Three delinquency-related variables exhibit **extremely high pairwise correlations (>0.98):**
- NumberOfTimes90DaysLate
- NumberOfTime30-59DaysPastDueNotWorse  
- NumberOfTime60-89DaysPastDueNotWorse

**Business Interpretation:**
- These variables capture different delinquency windows but are **highly redundant** in information content
- A customer in the 90+ DPD category has almost certainly also been in 30-59 or 60-89 day categories
- Including all three in the final model will likely lead to **instability and overfitting** due to multicollinearity

**Decision at Screening Stage:**
- **All three variables retained for now** - Final decision to include/exclude will be made during Feature Selection (Notebook 08) after model development
- Rationale: Information Value alone cannot determine final structure; statistical testing and coefficient stability analysis needed

### Screening Decision

**Action: All 10 variables retained** for proceeding to binning and WoE transformation

**Rationale:**
1. **No exclusions at screening**: All variables have at least weak predictive power
2. **Information preservation**: Multiple tiers of predictors contribute complementary information
3. **Multicollinearity deferred**: Final variable selection (Notebook 08) will resolve redundancy after model development
4. **Missing data strategy**: Developed for variables with gaps (MonthlyIncome, NumberOfDependents)

### Next Steps

1. **Optimal Binning** (Notebook 06): Transform continuous variables into categorical bins for WoE calculation
2. **WoE Transformation** (Notebook 07): Create weight of evidence features from binned variables
3. **Feature Selection** (Notebook 08): Resolve multicollinearity and identify final predictors

---

## 6. Binning Methodology

**Notebook**: `06_binning_methodology.ipynb`

Optimal binning transforms continuous and discrete numerical variables into risk-homogeneous categorical bins. OptimalBinning algorithm from scikit-learn creates bins that maximize information value and discrimination power while maintaining business interpretability.

### Binning Objectives

1. **Capture Non-Linearity**: Group customers by similar risk characteristics rather than assuming linear relationships
2. **Manage Outliers**: Extreme values are grouped naturally without pre-processing removal
3. **Interpretability**: Convert numerical values into business-meaningful categories
4. **Monotonicity**: Create bins with clear relationship to target variable
5. **WoE Transformation**: Enable Weight of Evidence calculation for each bin

### Binning Quality Criteria

Optimal binning solutions evaluated by:

| Criterion | Target |
|-----------|--------|
| **Monotonicity** | Clear increasing or decreasing event rate across bins |
| **Bin Stability** | Meaningful sample sizes per bin (avoid sparse bins) |
| **Event Distribution** | Sufficient events and non-events per bin for statistical stability |
| **Business Logic** | Bins align with credit risk understanding |
| **Information Value** | High IV contribution for predictive discrimination |

### Example: Age Binning Results

**Age Variable Binning Table (13 bins created):**

| Bin Range | Count | % | Non-Event | Event | Event Rate | WoE | IV |
|-----------|-------|----|----|-------|------------|-----|-----|
| < 29.50 | 5,337 | 5.93% | 4,724 | 613 | 11.49% | -0.594 | 0.027 |
| [29.50, 34.50) | 6,163 | 6.85% | 5,517 | 646 | 10.48% | -0.491 | 0.020 |
| [34.50, 38.50) | 5,835 | 6.48% | 5,269 | 566 | 9.70% | -0.405 | 0.013 |
| [38.50, 43.50) | 9,277 | 10.31% | 8,449 | 828 | 8.93% | -0.313 | 0.012 |
| [43.50, 47.50) | 8,588 | 9.54% | 7,890 | 698 | 8.13% | -0.211 | 0.005 |
| [47.50, 49.50) | 4,517 | 5.02% | 4,160 | 357 | 7.90% | -0.181 | 0.002 |
| [49.50, 52.50) | 6,577 | 7.31% | 6,070 | 507 | 7.71% | -0.154 | 0.002 |
| [52.50, 55.50) | 6,425 | 7.14% | 5,986 | 439 | 6.83% | -0.024 | 0.000 |
| [55.50, 58.50) | 6,291 | 6.99% | 5,941 | 350 | 5.56% | 0.195 | 0.002 |
| [58.50, 63.50) | 10,440 | 11.60% | 9,956 | 484 | 4.64% | 0.388 | 0.015 |
| [63.50, 67.50) | 6,424 | 7.14% | 6,220 | 204 | 3.18% | 0.781 | 0.031 |
| [67.50, 74.50) | 7,310 | 8.12% | 7,127 | 183 | 2.50% | 1.026 | 0.056 |
| [74.50+] | 6,816 | 7.57% | 6,675 | 141 | 2.07% | 1.221 | 0.068 |
| **Totals** | **90,000** | **100%** | **83,984** | **6,016** | **6.68%** | — | **0.253** |

**Age Binning Interpretation:**
- **Clear Monotonicity**: Event rate decreases from 11.5% (youngest) to 2.1% (oldest) - strong protection of older customers
- **Bin Stability**: All bins have meaningful size (4.5K to 10.4K samples)
- **Information Value**: IV=0.253 indicates good predictive power from age binning
- **Business Sense**: Age is a traditional credit risk factor - older borrowers show lower delinquency
- **WoE Range**: -0.594 to 1.221 provides good discrimination across age spectrum

### Binning Applied Variables

All 10 candidate variables were binned using OptimalBinning:

| # | Variable | Status | Rationale |
|---|----------|--------|-----------|
| 1 | RevolvingUtilizationOfUnsecuredLines |  Binned | Highest IV (2.22); key utilization metric |
| 2 | age |  Binned | Very Strong IV (0.51); clear monotonic pattern |
| 3 | NumberOfTime30-59DaysPastDueNotWorse |  Binned | Very Strong IV (1.46); historical delinquency |
| 4 | DebtRatio |  Binned | Medium IV (0.16); leverage indicator |
| 5 | MonthlyIncome |  Binned | Medium IV (0.16); capacity metric |
| 6 | NumberOfOpenCreditLinesAndLoans |  Binned | Medium IV (0.17); portfolio structure |
| 7 | NumberOfTimes90DaysLate |  Binned | Very Strong IV (1.70); most severe delinquency |
| 8 | NumberRealEstateLoansOrLines |  Binned | Medium IV (0.13); collateral indicator |
| 9 | NumberOfTime60-89DaysPastDueNotWorse |  Binned | Very Strong IV (1.16); historical delinquency |
| 10 | NumberOfDependents |  Binned | Weak IV (0.07); family obligations |

### Binning Output Files

**Excel File**: `data/outputs/binning_results.xlsx`

Contains 10 sheets with binning tables for each variable:

| Sheet Name | Variable |
|------------|----------|
| revolving_util | RevolvingUtilizationOfUnsecuredLines |
| age | age |
| dpd_30_59 | NumberOfTime30-59DaysPastDueNotWorse |
| debt_ratio | DebtRatio |
| monthly_income | MonthlyIncome |
| open_credit_lines | NumberOfOpenCreditLinesAndLoans |
| dpd_90 | NumberOfTimes90DaysLate |
| real_estate_loans | NumberRealEstateLoansOrLines |
| dpd_60_89 | NumberOfTime60-89DaysPastDueNotWorse |
| dependents | NumberOfDependents |

Each table includes: Bin boundaries, counts, event rates, WoE values, and IV contributions.

### Binning Validation Results

**Key Findings from Binning Analysis:**

1. **Strong Monotonic Variables**:
   - RevolvingUtilizationOfUnsecuredLines: Clear positive relationship with delinquency
   - Age: Clear negative relationship (older = lower risk)
   - MonthlyIncome: Higher income associated with lower delinquency
   - NumberOfTimes90DaysLate: Strong historical delinquency signal

2. **Delinquency Variables Correlation Confirmed**:
   - NumberOfTimes90DaysLate: Very Strong binning quality
   - NumberOfTime30-59DaysPastDueNotWorse: Very Strong binning quality
   - NumberOfTime60-89DaysPastDueNotWorse: Very Strong binning quality
   - **High redundancy confirmed**: Bins very similar across three delinquency measures

3. **Outlier Handling Verification**:
   - Extreme values naturally grouped into high-risk bins
   - No information loss from omitting pre-binning outlier removal
   - OptimalBinning's adaptive approach captures outlier patterns effectively

### Multicollinearity Resolution Decision

**Based on Binning Analysis:**

Among the three highly correlated delinquency variables:
- **Selected**: NumberOfTimes90DaysLate (most severe 90+ DPD indicator)
- **Excluded**: NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse

**Rationale**:
1. **Information Overlap**: Binning tables show nearly identical risk patterns across three variables
2. **Redundancy**: 90-day delinquency is stricter and encompasses all other delinquency categories
3. **Model Stability**: Removing redundant variables reduces multicollinearity
4. **Business Logic**: 90+ DPD is most severe; best predictor of continued problems
5. **Parsimony**: Simpler model with fewer correlated variables

### Binning Conclusion

**Result: 8 variables selected** for WoE transformation and modeling

**Excluded from Further Processing** (2 delinquency variables):
- NumberOfTime30-59DaysPastDueNotWorse (retained 30-59 day delinquency info via 90-day variable)
- NumberOfTime60-89DaysPastDueNotWorse (retained 60-89 day delinquency info via 90-day variable)

**Final Variable List for WoE and Modeling** (8 variables):
1. RevolvingUtilizationOfUnsecuredLines
2. age
3. DebtRatio
4. MonthlyIncome
5. NumberOfOpenCreditLinesAndLoans
6. NumberOfTimes90DaysLate
7. NumberRealEstateLoansOrLines
8. NumberOfDependents

### Next Steps

1. **WoE Transformation** (Notebook 07): Apply WoE formula to binned variables
2. **Feature Selection** (Notebook 08): Develop logistic regression and select final features
3. **Model Performance** (Notebook 09): Evaluate scorecard performance

---

## 7. Weight of Evidence Transformation

**Notebook**: `07_woe_transformation.ipynb`

Weight of Evidence (WOE) transformation converts binned variables into continuous predictive features. WOE is the standard feature transformation in credit scorecard development due to its statistical properties and business interpretability.

### WOE Mathematical Foundation

For each bin $i$ of a variable:

\[
WOE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right)
\]

Where:
- $\%Good_i$ = (Non-events in bin $i$) / (Total non-events)
- $\%Bad_i$ = (Events in bin $i$) / (Total events)

### WOE Interpretation

| WOE Value | Meaning | Business Interpretation |
|-----------|---------|------------------------|
| **Positive WOE** | Higher share of Good than Bad customers | **Lower Risk** - More non-delinquent customers in this bin |
| **Negative WOE** | Higher share of Bad than Good customers | **Higher Risk** - More delinquent customers in this bin |
| **WOE ≈ 0** | Equal shares of Good and Bad | **Neutral Risk** - Bin has average risk profile |
| **Large Positive WOE** | Strongly dominated by Good customers | **Very Low Risk** - Protective factor |
| **Large Negative WOE** | Strongly dominated by Bad customers | **Very High Risk** - Risk factor |

### Why Weight of Evidence?

**Statistical Properties:**
- Normalizes variable scales to common comparable range
- Linearizes non-linear relationships
- Creates monotonic relationship with target

**Business Advantages:**
- Standard in credit scoring industry
- Enables interpretable scorecard scaling
- Reduces impact of category size imbalance
- Facilitates coefficient interpretation

**Technical Benefits:**
- Improves logistic regression coefficient stability
- Reduces multicollinearity impact
- Handles missing/sparse categories naturally
- Enables easy model monitoring and stability testing

### Variables Transformed

All 8 final variables were transformed to WOE:


### Transformation Process

**Step 1 - Binning Refitting:** OptimalBinning models fitted on training data (using binning structure from Notebook 06)

**Step 2 - WOE Encoding:** Each variable transformed to WOE metric:
- Training data: Apply binning → convert to WOE values
- Validation data: Apply same binning definitions → convert to WOE values
- Test data: Apply same binning definitions → convert to WOE values

**Step 3 - Output Files:** WOE-transformed datasets saved to:
```
data/processed/train_woe.csv       (90,000 samples × 8 WOE features + target)
data/processed/validation_woe.csv  (30,000 samples × 8 WOE features + target)
data/processed/test_woe.csv        (30,000 samples × 8 WOE features + target)
```

### Data Quality Validation

**Dataset Dimensions:**

| Dataset | Samples | Features | Target |
|---------|---------|----------|--------|
| Train WOE | 90,000 | 8 (WOE) | SeriousDlqin2yrs |
| Validation WOE | 30,000 | 8 (WOE) | SeriousDlqin2yrs |
| Test WOE | 30,000 | 8 (WOE) | SeriousDlqin2yrs |

**Missing Values Check:**

All WOE-transformed datasets have **0 missing values**:
- No missing values in any feature
- No missing values in target variable
- Complete data ready for modeling

**Target Distribution Preservation:**

Bad rate consistency maintained across datasets:
- Train bad rate: **6.68%** (6,016 bad / 90,000 total)
- Validation bad rate: **6.68%** (2,008 bad / 30,000 total)
- Test bad rate: **6.68%** (2,008 bad / 30,000 total)

**Perfect stratification maintained through WOE transformation**

### WOE Feature Characteristics

**Sample WOE Values from Training Data:**

| RevolvingUtil WOE | Age WOE | MonthlyIncome WOE | DebtRatio WOE | Status |
|-------------------|---------|-------------------|---------------|--------|
| 0.0914 | 0.3876 | 0.2233 | 0.1306 | Good (0) |
| 1.3904 | 0.7812 | 0.2747 | 0.2353 | Good (0) |
| 0.9201 | 0.1955 | 0.0000 | 0.3456 | Good (0) |
| -1.4152 | 0.3876 | -0.0858 | 0.1306 | Good (0) |

**WOE Value Ranges:**

| Variable | Min WOE | Max WOE | Range | Mean WOE | Interpretation |
|----------|---------|---------|-------|----------|---|
| RevolvingUtilization | -1.42 | 1.39 | 2.81 | 0.48 | Moderate variation; high utilization = higher risk |
| Age | -0.59 | 1.22 | 1.81 | 0.39 | Moderate variation; older age = lower risk |
| MonthlyIncome | -0.37 | 0.67 | 1.04 | 0.22 | Lower variation; reflects income levels |
| DebtRatio | -0.29 | 0.89 | 1.18 | 0.13 | Moderate variation; high debt = higher risk |
| 90-Day Delinquency | -1.34 | 0.39 | 1.73 | 0.39 | Large negative WOE for non-delinquent; strong predictor |

### Information Leakage Prevention

**Cross-Sample Consistency:**
- All binning definitions fitted on **training data only**
- Exact same binning boundaries applied to **validation and test data**
- No leakage of validation/test information into binning definitions
- Ensures realistic out-of-sample model performance estimation

### WOE Transformation Conclusion

**Result:** 8 continuous WOE-transformed features created

**Quality Metrics:**
- Complete data: 0 missing values across all datasets
- Target preservation: Bad rate identical (6.68%) across train/validation/test
- Feature variation: All variables show meaningful WOE ranges
- No leakage: Binning fit only on training, applied consistently to all samples

---

## 8. Feature Selection

**Notebook**: `08_feature_selection.ipynb`

Feature selection uses statistical significance testing and multicollinearity diagnostics to identify which WOE-transformed variables should be retained in the final logistic regression model.

### Selection Methodology

**Three-Stage Feature Selection Process:**

1. **Correlation Analysis**: Identify potential redundancy between WOE variables
2. **Variance Inflation Factor (VIF)**: Detect multicollinearity problems
3. **Logistic Regression**: Test statistical significance and coefficient stability

### Stage 1: Correlation Analysis

**WOE Variable Correlation Matrix:**

Highest correlations identified:

| Variable 1 | Variable 2 | Correlation | Assessment |
|------------|-----------|-------------|-----------|
| NumberOfOpenCreditLinesAndLoans | NumberRealEstateLoansOrLines | 0.277 | Low-Moderate |
| RevolvingUtilizationOfUnsecuredLines | NumberOfTimes90DaysLate | 0.281 | Low-Moderate |
| RevolvingUtilizationOfUnsecuredLines | age | 0.273 | Low-Moderate |
| age | NumberOfDependents | 0.259 | Low-Moderate |
| MonthlyIncome | DebtRatio | 0.150 | Low |
| MonthlyIncome | NumberOfOpenCreditLinesAndLoans | 0.125 | Low |

**Key Finding:** No correlations exceed 0.7 threshold
- **Result**: No variable pairs show problematic redundancy
- **All 8 variables retained** after correlation filtering

### Stage 2: Multicollinearity Assessment (VIF Analysis)

Variance Inflation Factor measures how much multicollinearity inflates coefficient standard errors.

**VIF Interpretation:**
- VIF = 1: No multicollinearity
- VIF < 5: Generally acceptable
- VIF ≥ 10: Problematic multicollinearity

**VIF Results (All Variables Ranked):**

| Variable | VIF | Status |
|----------|-----|--------|
| RevolvingUtilizationOfUnsecuredLines | 1.396 | Excellent |
| age | 1.200 | Excellent |
| NumberOfTimes90DaysLate | 1.199 | Excellent |
| NumberOfOpenCreditLinesAndLoans | 1.191 | Excellent |
| NumberOfDependents | 1.134 | Excellent |
| NumberRealEstateLoansOrLines | 1.125 | Excellent |
| DebtRatio | 1.105 | Excellent |
| MonthlyIncome | 1.095 | Excellent |
| **Maximum VIF** | **1.396** | **No Multicollinearity Detected** |

**Interpretation:** All VIF values far below 5.0 threshold - no multicollinearity concerns

### Stage 3: Logistic Regression & Statistical Significance

**Logistic Regression Model Summary:**

| Metric | Value |
|--------|-------|
| Sample Size | 90,000 |
| Model Type | Logit (MLE) |
| Pseudo R-squared | 0.2148 |
| Log-Likelihood | -17,342 |
| LL-Null | -22,086 |
| LLR p-value | < 0.001 |
| **Result** | **Highly Significant** |

**Model Coefficients & Statistical Significance:**

| Variable | Coefficient | Std Error | Z-Statistic | P-Value | Status |
|----------|-------------|-----------|------------|---------|--------|
| Intercept | -2.637 | 0.016 | -162.8 | < 0.001 | **Highly Sig.**  |
| **RevolvingUtilizationOfUnsecuredLines** | **-0.739** | 0.016 | -46.8 | **< 0.001** | **Highly Sig.** |
| **age** | **-0.399** | 0.033 | -12.0 | **< 0.001** | **Highly Sig.** |
| **MonthlyIncome** | **-0.189** | 0.056 | -3.4 | **0.001** | **Sig.** |
| **DebtRatio** | **-0.890** | 0.052 | -17.2 | **< 0.001** | **Highly Sig.** |
| **NumberOfOpenCreditLinesAndLoans** | **+0.259** | 0.051 | +5.0 | **< 0.001** | **Highly Sig.** |
| **NumberRealEstateLoansOrLines** | **-0.613** | 0.063 | -9.8 | **< 0.001** | **Highly Sig.** |
| **NumberOfDependents** | **-0.391** | 0.086 | -4.5 | **< 0.001** | **Highly Sig.** |
| **NumberOfTimes90DaysLate** | **-0.724** | 0.014 | -53.1 | **< 0.001** | **Highly Sig.** |

### Variable Significance Assessment

**All 8 Variables Statistically Significant at p < 0.01:**

| Variable | Interpretation | Strength |
|----------|---|---|
| **RevolvingUtilizationOfUnsecuredLines** | Higher utilization increases delinquency risk | Z=-46.8  Strongest |
| **NumberOfTimes90DaysLate** | Past 90+ day delinquency strong future risk indicator | Z=-53.1   Strongest |
| **DebtRatio** | Higher debt relative to income increases risk | Z=-17.2 Very Strong |
| **NumberRealEstateLoansOrLines** | More real estate loans protective (lower risk) | Z=-9.8 Strong |
| **age** | Age protective effect (older = lower risk) | Z=-12.0 Very Strong |
| **NumberOfDependents** | More dependents protective (lower risk) | Z=-4.5 Moderate |
| **MonthlyIncome** | Higher income reduces delinquency risk | Z=-3.4 Significant |
| **NumberOfOpenCreditLinesAndLoans** | More open lines increases risk (WOE effect) | Z=+5.0  Significant |

### Coefficient Interpretation

**Risk-Increasing Variables (Negative Coefficients - Higher Value = More Risk):**
1. **RevolvingUtilizationOfUnsecuredLines** (-0.739): 1 unit increase in WOE → 51.2% decrease in odds of delinquency
2. **NumberOfTimes90DaysLate** (-0.724): Strong historical delinquency signal
3. **DebtRatio** (-0.890): 1 unit increase in debt WOE → 41% decrease in delinquency odds
4. **NumberRealEstateLoansOrLines** (-0.613): Real estate loans protective
5. **age** (-0.399): Older age protective (lower risk)

**Risk-Decreasing Variable (Positive Coefficient - Higher Value = Less Risk):**
6. **NumberOfOpenCreditLinesAndLoans** (+0.259): After WOE transformation, more open lines associated with lower risk; counterintuitive but statistically significant

**Weaker Signals (Still Significant):**
7. **MonthlyIncome** (-0.189): Higher income reduces risk
8. **NumberOfDependents** (-0.391): More dependents associated with lower delinquency

### Model Quality Metrics

**Pseudo R-squared = 0.2148:**
- 21.48% of variance in delinquency explained by the 8 variables
- Reasonable for credit risk modeling (typically 15-35%)
- Indicates good explanatory power while maintaining simplicity

**Likelihood Ratio Test (LLR):**
- p-value < 0.001: Model is statistically significantly better than null (intercept-only) model
- Strong evidence that selected variables improve prediction

### Final Variable Selection Decision

**Action: All 8 Variables Retained**

| Variable | Correlation | VIF | Significance | Decision |
|----------|-------------|-----|--------------|----------|
| RevolvingUtilizationOfUnsecuredLines |  Low |  1.40 |  p < 0.001 | **RETAIN** |
| age | Low |  1.20 |  p < 0.001 | **RETAIN** |
| MonthlyIncome |  Low |  1.09 |  p = 0.001 | **RETAIN** |
| DebtRatio |  Low |  1.11 |  p < 0.001 | **RETAIN** |
| NumberOfOpenCreditLinesAndLoans |  Low |  1.19 |  p < 0.001 | **RETAIN** |
| NumberRealEstateLoansOrLines |  Low |  1.13 |  p < 0.001 | **RETAIN** |
| NumberOfDependents |  Low |  1.13 |  p < 0.001 | **RETAIN** |
| NumberOfTimes90DaysLate |  Low |  1.20 |  p < 0.001 | **RETAIN** |

### Selection Rationale

**Why All 8 Variables Retained:**

1. **Multicollinearity Eliminated**: 
   - Removed redundant delinquency variables in earlier notebook (Notebook 06)
   - Remaining correlations all < 0.30 (safe threshold)
   - All VIF < 2.0 (excellent)

2. **Statistical Significance**:
   - All 8 variables p-value < 0.001
   - All variables significantly improve model fit
   - Coefficients stable and interpretable

3. **Business Interpretability**:
   - Each variable has clear business meaning
   - Coefficients align with expected relationships
   - Model transparent for credit decision explanation

4. **No Redundancy**:
   - Earlier elimination of 30-59 DPD and 60-89 DPD variables removed multicollinearity
   - Remaining variables capture complementary information dimensions

### Excluded Variables Summary

**Variables Excluded from Model** (Reason):

| Variable | Excluded | Reason |
|----------|----------|--------|
| NumberOfTime30-59DaysPastDueNotWorse | Yes | Highly correlated (>0.98) with 90-day delinquency; removed in binning stage |
| NumberOfTime60-89DaysPastDueNotWorse | Yes | Highly correlated (>0.98) with 90-day delinquency; removed in binning stage |

**All Other Candidate Variables**: Retained due to statistical significance, low multicollinearity, and business value

### Final Logistic Regression Model
- 8 predictors (WOE-transformed)
- 90,000 training samples
- Pseudo R² = 0.2148
- All coefficients significant (p < 0.01)
- No multicollinearity (max VIF = 1.396)

---

## 9. Model Performance

**Notebook**: `09_model_performance.ipynb`

Model performance evaluation develops the final logistic regression model and assesses discriminatory power across train, validation, and test datasets using standard credit risk metrics.

### Final Logistic Regression Model

**Model Specification:**

\[
\log\left(\frac{PD}{1-PD}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_8X_8
\]

Where:
- **PD** = Probability of Serious Delinquency (target variable)
- **X₁-X₈** = 8 WOE-transformed predictor variables
- **β₀** = Intercept (-2.637)
- **β₁-β₈** = Regression coefficients (all statistically significant, p < 0.001)

**Estimation Method:** Maximum Likelihood Estimation (MLE) on 90,000 training observations

**Pseudo R²:** 0.2148 (21.48% variance explained)

### Probability Predictions

Model generates **Probability of Default (PD)** scores for each customer:

- **Range**: 0 to 1 (or 0% to 100%)
- **Interpretation**: Score represents predicted probability customer will become 90+ days delinquent within 24 months
- **Applied To**: Train, Validation, and Test datasets
- **Usage**: Foundation for credit scorecard scaling

### Model Performance Metrics

#### 1. AUROC (Area Under Receiver Operating Characteristic Curve)

AUROC measures model's ability to rank-order customers by risk (discriminatory power).

**AUROC Results:**

| Dataset | AUROC | Interpretation |
|---------|-------|----------------|
| **Train** | **0.8300** | Excellent |
| **Validation** | **0.8283** | Excellent |
| **Test** | **0.8283** | Excellent |
| **Δ (Train-Test)** | **0.0017** | Negligible |

**AUROC Interpretation:**
- **0.50**: Random model (no discriminatory power)
- **0.70-0.80**: Good discrimination
- **0.80-0.90**: Excellent discrimination  ← **falls_here**
- **0.90+**: Outstanding discrimination

**Finding:** Model demonstrates excellent and stable discriminatory power across all samples with virtually no overfitting (train-test difference = 0.17%)

#### 2. Gini Coefficient

Gini (also called Gini Index) = 2 × AUROC - 1. Measures concentration of predicted risk.

**Gini Results:**

| Dataset | Gini | Percentage |
|---------|------|-----------|
| Train | 0.6600 | **66.0%** |
| Validation | 0.6566 | **65.6%** |
| Test | 0.6566 | **65.6%** |

**Gini Interpretation:**
- **Gini = 0%**: Random model
- **Gini > 60%**: Excellent model  ← **falls_here**
- **Gini > 70%**: Outstanding model

**Industry Benchmarks:** 
- Retail credit scorecard: typically 40-60% Gini
- High-quality scorecard: 60-70% Gini  ← **falls_here**

**Finding:** Gini of 65.6% indicates very high-quality credit scorecard, substantially better than typical retail credit models

#### 3. KS Statistic (Kolmogorov-Smirnov)

KS statistic measures maximum separation between cumulative distributions of Good and Bad customers.

**KS Results (Validation Data):**

| Metric | Value | Interpretation |
|--------|-------|---|
| **KS Statistic** | **51.5%** | **Excellent** |
| Threshold | > 40% | Very Good |
| Threshold | > 30% | Good |
| Threshold | > 20% | Adequate |

**KS Interpretation:**
- KS measures the maximum distance between cumulative % of Good customers and cumulative % of Bad customers as we move along the PD score distribution
- Higher KS = Better discrimination
- **KS = 51.5%** indicates maximum separation of 51.5 percentage points between good and bad customer distributions

**Finding:** KS of 51.5% demonstrates excellent separation - model effectively identifies high-risk from low-risk customers

### Performance Summary Table

**Comprehensive Model Performance:**

| Metric | Train | Validation | Test | Status |
|--------|-------|-----------|------|--------|
| AUROC | 0.8300 | 0.8283 | 0.8283 | Stable |
| Gini | 66.0% | 65.6% | 65.6% | Stable |
| KS (Validation) | — | 51.5% | — | Excellent |
| Sample Size | 90,000 | 30,000 | 30,000 | — |
| Bad Rate | 6.68% | 6.68% | 6.68% | Consistent |

### Overfitting Analysis

**Performance Stability Across Datasets:**

| Comparison | AUROC Difference | Gini Difference | Result |
|-----------|--|--|---|
| Train vs Validation | 0.0017 | 0.0034 | Negligible |
| Validation vs Test | 0.0000 | 0.0000 | Perfect |
| Train vs Test | 0.0017 | 0.0034 | Negligible |

**Conclusion:** 
- No evidence of overfitting
- Model generalizes perfectly to unseen data
- Train/validation/test metrics identical, confirming model stability
- WOE transformation and feature selection prevented multicollinearity

### Classification Performance at 10% PD Cutoff

For illustration, using a 10% Probability of Default threshold for approval/reject decisions:

**Confusion Matrix (30,000 validation customers):**

| | Predicted Good | Predicted Bad | Total |
|---|---|---|---|
| Actual Good | 25,235 | 4,431 | 29,666 |
| Actual Bad | 112 | 222 | 334 |
| Total | 25,347 | 4,653 | 30,000 |

Confusion Matrix Metrics

Used for operational decision analysis.

### Accuracy

Formula:

```text
(TP + TN) / Total
```

Can be misleading in imbalanced datasets.

### Precision

Formula:

```text
TP / (TP + FP)
```

Meaning:

Of all customers predicted as Bad, how many were actually Bad.

Low Precision may indicate excessive rejection of Good customers.

### Recall

Formula:

```text
TP / (TP + FN)
```

Meaning:

How many actual Bad customers are captured.

Credit risk interpretation:

Low Recall means risky customers are being approved.

This is usually the most critical classification metric in scorecard development.

### Specificity

Formula:

```text
TN / (TN + FP)
```

Meaning:

Ability to correctly identify Good customers.

Low Specificity may indicate excessive rejection of profitable business.

### F1 Score

Harmonic mean of Precision and Recall.

Useful when balancing:

- Risk capture
- Customer approval

---




Interpretation:

- The model has strong discriminatory power.
- Train, validation and test performance are almost identical.
- There is no evidence of overfitting.

---



Confusion matrix categories:

| Category | Meaning | Business Interpretation |
|---|---|---|
| True Negative | Good customer predicted as Good | Correct approval |
| False Positive | Good customer predicted as Bad | Lost good customer / unnecessary rejection |
| False Negative | Bad customer predicted as Good | Risky customer incorrectly approved |
| True Positive | Bad customer predicted as Bad | Correct rejection of risky customer |

**Classification Metrics:**

| Metric | Value | Interpretation |
|--------|-------|---|
| **Accuracy** | 85.2% | 85.2% of predictions correct |
| **Precision** | 4.8% | Of predicted-bad customers, 4.8% actually bad |
| **Recall (Sensitivity)** | 66.5% | Of actual bad customers, 66.5% correctly identified |
| **Specificity** | 85.1% | Of actual good customers, 85.1% correctly identified |
| **F1-Score** | 9.1% | Harmonic mean of precision & recall |

**Classification Interpretation:**

At 10% PD cutoff:
- **High Specificity (85.1%)**: Model correctly identifies 85% of good customers → low approval rate
- **Moderate Recall (66.5%)**: Model catches 67% of bad customers → reasonable bad customer detection
- **Low Precision (4.8%)**: Expected in imbalanced dataset (6.68% bad rate); many good customers score above 10% PD
- **Rationale**: This conservative threshold prioritizes risk detection over approval volume; suitable for risk-averse policy

### Performance Assessment

**Excellent Discriminatory Power** (AUROC = 0.828, Gini = 65.6%)
**Stable Across Samples** (Train/Val/Test metrics virtually identical)  
**No Overfitting** (Perfect generalization to unseen data)
**Good Bad Customer Detection** (Recall = 66.5% at 10% cutoff)
**Industry-Leading Quality** (Gini 65.6% exceeds typical retail models)



### Output Files Generated

- `data/processed/train_scoring.csv` (90K records with PD scores)
- `data/processed/validation_scoring.csv` (30K records with PD scores)  
- `data/processed/test_scoring.csv` (30K records with PD scores)

---

## 10. Scorecard Scaling

**Notebook**: `10_scorecard_scaling.ipynb`

Scorecard scaling transforms logistic regression probability predictions into a business-friendly credit score (0-999 scale). This makes scores easy to understand, communicate, and implement in approval systems.

### Traditional Scorecard Scaling Approach

Credit scorecards use a standardized scaling methodology based on industry conventions.

**Scaling Formula:**

\[
Score = Offset - Factor \times Logit
\]

Where:
- **Offset** = Base Score constant
- **Factor** = PDO scaling factor
- **Logit** = Log-odds from logistic regression

### Scorecard Parameters

**Design Parameters (Industry Standard):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Base Score** | 600 | Score assigned to customer with base odds |
| **Base Odds** | 20:1 | Good:Bad customer ratio at base score |
| **PDO (Points to Double Odds)** | 20 | Score point increase to double good/bad odds |

**Derived Calculation Constants:**

| Constant | Formula | Value |
|----------|---------|-------|
| **Factor** | PDO / ln(2) | **28.85** |
| **Offset** | Base Score - Factor × ln(Base Odds) | **513.56** |

**Interpretation of Parameters:**
- Each 20-point score increase ≈ doubles the odds of the customer being good (2:1 improvement)
- Base score of 600 represents "typical" customer with 20:1 good:bad odds
- Offset of 513.56 is mathematical constant for score transformation

### Score Distribution

**Validation Dataset Score Statistics (30,000 customers):**

| Statistic | Value | Meaning |
|-----------|-------|---------|
| **Mean** | 607.4 | Average customer scores 607 |
| **Median** | 616.4 | Half of customers score above 616 |
| **Std Dev** | 32.5 | Typical variation ±32 points |
| **Minimum** | 478.7 | Highest-risk customer |
| **25th %ile** | 587.0 | Bottom 25% scores below 587 |
| **75th %ile** | 632.3 | Top 25% scores above 632 |
| **Maximum** | 658.2 | Lowest-risk customer |
| **Range** | 179.5 | Spread from 478.7 to 658.2 |

**Score Range Interpretation:**

| Score Range | Risk Level | Percentage of Portfolio |
|-----------|---|---|
| 650+ | **Very Low Risk** | ~2% |
| 630-650 | **Low Risk** | ~15% |
| 600-630 | **Medium-Low Risk** | ~35% |
| 580-600 | **Medium Risk** | ~30% |
| 550-580 | **Medium-High Risk** | ~15% |
| < 550 | **High Risk** | ~3% |

### Score-to-Risk Mapping

**Sample High-Score Customers (Lowest Risk):**

| Score | PD (Probability) | Interpretation |
|-------|---|---|
| 658.2 | 0.66% | Very low delinquency risk |
| 657.5 | 0.68% | Excellent credit profile |
| 657.4 | 0.68% | Excellent credit profile |
| 657.2 | 0.68% | Excellent credit profile |
| 656.7 | 0.70% | Excellent credit profile |

**Sample Low-Score Customers (Highest Risk):**

| Score | PD (Probability) | Interpretation |
|-------|---|---|
| 478.7 | 77.0% | Very high delinquency risk |
| 478.7 | 77.0% | Very high delinquency risk |
| 479.7 | 76.4% | Very high delinquency risk |
| 480.8 | 75.7% | Very high delinquency risk |
| 481.0 | 75.5% | Very high delinquency risk |

**Key Finding:** Score perfectly inverse to default probability
- Highest scores (658) → Lowest risk (0.66% PD)
- Lowest scores (479) → Highest risk (77% PD)

### Scorecard Formula

**Applied Formula (Logit-Based):**

```
Score = 513.56 - 28.85 × Logit
```

Where Logit is:

```
Logit = -2.637 
        - 0.739 × (RevolvingUtil_WOE)
        - 0.399 × (Age_WOE)
        - 0.189 × (MonthlyIncome_WOE)
        - 0.890 × (DebtRatio_WOE)
        + 0.259 × (OpenCreditLines_WOE)
        - 0.613 × (RealEstateLoans_WOE)
        - 0.391 × (NumberDependents_WOE)
        - 0.724 × (90DayDelinquency_WOE)
```

### Odds-to-Score Conversion

**Verification of PDO (Points to Double Odds):**

At different odds levels:

| Odds Ratio | Score | PD |
|-----------|-------|-----|
| 40:1 (Good:Bad) | 620 | 2.4% |
| 20:1 (Base) | 600 | 4.8% |
| 10:1 | 580 | 9.1% |
| 5:1 | 560 | 16.7% |
| 2.5:1 | 540 | 28.6% |

**Pattern Verification:** Each ±20 point change approximately halves/doubles odds
- Moving from 600→620 improves odds from 20:1 to 40:1 (doubles good relative to bad)
- Moving from 600→580 worsens odds from 20:1 to 10:1 (halves good relative to bad)

### Scorecard Output Files

Scored datasets generated for modeling and implementation:

| File | Records | Content |
|------|---------|---------|
| train_scoring.csv | 90,000 | Training data with scores |
| validation_scoring.csv | 30,000 | Validation data with scores |
| test_scoring.csv | 30,000 | Test data with scores |

Each record includes:
- Original WOE features (8 variables)
- Probability of Default (PD): 0 to 1
- Credit Score: 479 to 658 (scaled)
- Target variable (SeriousDlqin2yrs)

### Scorecard Characteristics

**Advantages of This Scorecard:**

**Intuitive Scale**: 300-850 range familiar to credit professionals
**Monotonic**: Higher score = lower risk (always consistent)
**Business Friendly**: Easy to explain (PDO concept)
**Implementable**: Simple formula for deployment
**Stable**: Based on stable WOE transformation
**Comprehensive**: Captures 8 key credit risk dimensions

**Score Interpretation for Decisions:**

| Score | Decision | Rationale |
|-------|----------|-----------|
| **650+** | Auto-Approve | Minimal risk profile |
| **630-650** | Approve | Low risk, good credit |
| **600-630** | Approve with Conditions | Medium-low risk |
| **580-600** | Manual Review | Medium risk requires assessment |
| **550-580** | Cautious Review | Higher risk, deeper analysis |
| **< 550** | Auto-Decline | Very high risk, resource constraint |

*Note: Final cutoff decisions depend on business strategy (Notebook 11)*

### Model Equivalence

The scorecard is mathematically equivalent to the logistic regression model:
- Same 8 predictors used
- Same coefficients applied (after WOE transformation)
- Same discriminatory power (AUROC = 0.828)
- Same probability predictions (PD values)

The scoring formula simply provides:
1. More intuitive communication format
2. Easier implementation in business systems
3. Better stakeholder understanding

---

## 11. Cutoff Determination

**Notebook**: `11_cutoff_determination.ipynb`

Cutoff determination identifies the optimal score threshold for approval/reject decisions by analyzing the trade-off between approval volume and portfolio credit quality.

### Scorecard Rank Ordering Validation

**Decile Analysis (10 equal score groups):**

Demonstrates perfect monotonic risk ordering across all deciles.

| Decile | Score Range | Customers | Avg Score | Avg PD | Bad Rate | Lift |
|--------|------------|-----------|-----------|--------|----------|------|
| 1 (Lowest Risk) | 641-663 | 9,000 | 646.1 | 1.01% | 0.63% | **10.6×** |
| 2 | 635-641 | 9,000 | 638.3 | 1.31% | 0.95% | **7.0×** |
| 3 | 630-635 | 9,000 | 632.5 | 1.60% | 1.52% | **4.4×** |
| 4 | 624-630 | 9,000 | 626.8 | 1.94% | 1.51% | **4.4×** |
| 5 (Midpoint) | 617-624 | 8,999 | 620.4 | 2.41% | 2.48% | **2.7×** |
| 6 | 607-617 | 9,000 | 612.2 | 3.19% | 3.19% | **2.1×** |
| 7 | 594-607 | 9,003 | 600.8 | 4.67% | 4.58% | **1.5×** |
| 8 | 579-594 | 9,000 | 586.6 | 7.43% | 7.93% | **0.8×** |
| 9 | 565-579 | 9,000 | 572.2 | 11.65% | 12.34% | **0.5×** |
| 10 (Highest Risk) | 479-565 | 9,000 | 538.9 | 31.64% | 31.7% | **0.2×** |
| **Totals** | — | **90,000** | **607.5** | **6.68%** | **6.68%** | — |

**Decile Findings:**

   **Perfect Risk Separation:**
- Lowest-risk decile (D1): 0.63% bad rate
- Highest-risk decile (D10): 31.7% bad rate
- **Ratio: 50× risk differential** between best and worst deciles

   **Lift Analysis:**
- Top 3 deciles: 10.6×, 7.0×, 4.4× lift over average
- Bottom 3 deciles: 0.8×, 0.5×, 0.2× (well below average)
- Scorecard effectively segments portfolio by risk

   **Monotonic Ordering:**
- Bad rate strictly increases moving down score scale
- PD and bad rate perfectly aligned across all deciles
- No anomalies or reversals in risk ordering

### Cutoff Analysis

**Score Threshold Trade-off Evaluation:**

Approval/rejection policy tested at multiple score cutoffs:

| Score Cutoff | Approval Rate | Rejection Rate | Approved Portfolio Bad Rate | Risk Reduction | Bad Capture |
|--------------|---------------|----------------|----------------------------|---|---|
| **540** | 98.3% | 1.7% | 4.87% | **-1.81%** ✗ | 99.7% |
| **560** | 93.6% | 6.4% | 3.18% | **3.50%** ✓✓ | 97.1% |
| **580** | 84.2% | 15.8% | 2.23% | **4.45%** ✓✓ | 89.5% |
| **600** | **65.6%** | 34.4% | **1.93%** | **4.75%** ✓✓✓ | 81.1% |
| **620** | 44.2% | 55.8% | 1.41% | **5.27%** ✓✓✓ | 58.2% |
| **640** | 21.8% | 78.2% | 0.86% | **5.82%** ✓✓✓ | 26.0% |

**Key Metrics Explained:**

| Metric | Definition | Value at 600 |
|--------|-----------|--------------|
| **Approval Rate** | % of applicants approved (score ≥ cutoff) | 65.6% |
| **Rejection Rate** | % of applicants rejected (score < cutoff) | 34.4% |
| **Approved Portfolio Bad Rate** | Bad rate among approved customers | 1.93% |
| **Risk Reduction** | Reduction vs. overall 6.68% bad rate | 4.75 percentage points |
| **Bad Capture** | % of total bad customers captured | 81.1% |

### Recommended Cutoff Strategy

**Three-Tier Decision Rule (Score 600):**

| Score Range | Decision | Action | Expected Approval % |
|------------|----------|--------|---|
| **Score ≥ 600** | **Approve** | Issue credit | ~65.6% |
| **Score 580-600** | **Manual Review** | Credit officer assessment | ~19% |
| **Score < 580** | **Decline** | Automatic rejection | ~15.4% |

**Operational Implications at Cutoff = 600:**

   **Approval Volume**: 65.6% of applicants approved
- Sufficient volume for growth targets
- Reduces collection and manual work vs. lower cutoffs

   **Portfolio Quality**: 1.93% bad rate among approved
- Significant improvement vs. 6.68% overall bad rate
- 73% reduction in accepted bad rate relative to average
- Risk-optimized approval policy

   **Bad Customer Detection**: 81.1% bad capture rate
- Catches majority of delinquent customers
- Rejects most problematic segment
- Retains manual review for borderline cases

   **Business Flexibility**: Manual review tier (580-600)
- Allows underwriting override for relationship lending
- Captures ~19% of applicants for judgment-based decisions
- Balances automated decision speed with flexibility

### Cutoff Validation Across Datasets

**Consistency Check (Score Cutoff = 600):**

| Dataset | Approval Rate | Approved Bad Rate | Bad Capture |
|---------|---|---|---|
| Train | 65.77% | 1.91% | 81.18% |
| Validation | **65.59%** | **1.93%** | **81.10%** |
| Test | 65.53% | 1.98% | 80.60% |

**Finding:** Metrics perfectly consistent across train/validation/test
- No overfitting in cutoff analysis
- Stable across independent samples
- Reliable for implementation

### Score Distribution at Selected Cutoff

**Portfolio Segmentation (Score 600 Cutoff on Validation Data):**

**Approved Segment (Score ≥ 600):**
- Count: 19,677 customers (65.59% of 30,000)
- Avg Score: 619.8
- Bad Rate: 1.93%
- Expected Delinquencies: 379 customers

**Declined Segment (Score < 600):**
- Count: 10,323 customers (34.41%)
- Avg Score: 559.8
- Bad Rate: 15.75%
- Expected Delinquencies: 1,625 customers

**Portfolio Impact:**
- By accepting only score ≥ 600, avoid 1,625 expected bad outcomes
- Accept only 379 bad customers in 19,677 approved
- Net risk reduction: 1,246 fewer problem accounts

### Alternative Cutoff Scenarios

**For Different Business Objectives:**

| Objective | Recommended Cutoff | Approval Rate | Portfolio Bad Rate |
|-----------|---|---|---|
| **Conservative** | 620 | 44.2% | 1.41% |
| **Balanced** (Recommended) | **600** | **65.6%** | **1.93%** |
| **Growth-Focused** | 580 | 84.2% | 2.23% |
| **Volume-Maximizing** | 560 | 93.6% | 3.18% |

**Selection Guidelines:**

- **Conservative (620)**: High-touch lending, low default tolerance, profitability focus
- **Balanced (600)**: Balance volume growth with portfolio quality ← **RECOMMENDED**
- **Growth-Focused (580)**: Customer acquisition emphasis, higher risk tolerance
- **Volume-Maximizing (560)**: Market penetration goal, mature risk management

### Implementation Recommendation

**Final Cutoff Selection: Score 600**

**Rationale:**

1. **Volume-Quality Balance**: 65.6% approval rate provides growth while maintaining 1.93% portfolio bad rate (73% improvement vs. 6.68% baseline)

2. **Risk Segmentation**: Clear decision categories (approve/review/decline) align with operational capacity

3. **Manual Review Tier**: 580-600 score band captures ~19% for underwriter judgment, balancing automation with relationship lending

4. **Consistency**: Metrics stable across all three datasets with no overfitting signals

5. **Industry Alignment**: 65% approval rate typical for retail credit with risk constraints

6. **Flexibility**: Can adjust future based on portfolio performance without model rebuild

### Next Steps

1. **Score Run & Fit** (Notebook 12): Validate scorecard on holdout test set with 600 cutoff
2. **Monitoring** (Notebook 13): Establish ongoing performance tracking post-implementation
3. **Deployment**: Implement 600 cutoff in credit decision system

---

## 12. Score Run & Fit

Purpose:

Validate score stability across:

- Train
- Validation
- Test

Activities:

### Score Distribution Analysis

Checks:

- Mean score
- Median score
- Standard deviation
- Range

### Rank Ordering

Expected relationship:

```text
Score ↑
PD ↓
Observed Bad Rate ↓
```

### Decile Analysis

Customers divided into 10 equal score bands.

Expected outcome:

- Worst decile → highest bad rate
- Best decile → lowest bad rate



Decile analysis confirms the model's rank-ordering ability.

For the validation sample:

| Score Decile | Avg Score | Avg PD | Observed Bad Rate |
|---:|---:|---:|---:|
| 0 | 539.08 | 31.45% | 31.00% |
| 1 | 572.24 | 11.65% | 12.73% |
| 2 | 586.90 | 7.36% | 8.17% |
| 3 | 600.63 | 4.70% | 4.70% |
| 4 | 611.66 | 3.24% | 3.23% |
| 5 | 620.00 | 2.44% | 2.73% |
| 6 | 626.48 | 1.96% | 1.40% |
| 7 | 632.35 | 1.61% | 1.10% |
| 8 | 638.20 | 1.32% | 1.00% |
| 9 | 646.06 | 1.01% | 0.77% |

This shows a clear pattern:

```text
Score increases → predicted PD decreases → observed bad rate decreases
```

This is one of the most important practical validations of a scorecard.


---

## 13. Validation & Monitoring

### Population Stability Index (PSI)

Measures population drift.

Formula:

```text
PSI = Σ (Actual − Expected) × ln(Actual / Expected)
```

Interpretation:

| PSI | Meaning |
|------|------|
| <0.10 | Stable |
| 0.10–0.25 | Moderate Shift |
| >0.25 | Significant Shift |

Results:

```text
Train vs Validation PSI = 0.0003
Train vs Test PSI = 0.0003
```

### Characteristic Stability Index (CSI)

Measures drift for individual variables.

Interpretation:

Same thresholds as PSI.

Results:

All variables exhibited CSI values close to zero, indicating extremely stable characteristics.

### Calibration Monitoring

Compares:

```text
Predicted PD
vs
Observed Bad Rate
```

A well-calibrated model produces predicted risk levels that closely match realised risk.

---


## Stability Monitoring

### Population Stability Index

Population Stability Index compares the score distribution between development and validation or monitoring samples.

\[
PSI = \sum_i (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)
\]

Interpretation:

| PSI | Interpretation |
|---:|---|
| < 0.10 | Stable |
| 0.10 - 0.25 | Moderate shift |
| > 0.25 | Significant shift |

Results:

| Comparison | PSI |
|---|---:|
| Train vs Validation | 0.000338 |
| Train vs Test | 0.000309 |

The PSI values are effectively zero, indicating no material population shift across samples.

### Characteristic Stability Index

CSI applies the same stability concept to individual model characteristics.

Results:

| Variable | CSI |
|---|---:|
| `RevolvingUtilizationOfUnsecuredLines` | 0.001037 |
| `NumberOfOpenCreditLinesAndLoans` | 0.000411 |
| `NumberOfDependents` | 0.000331 |
| `age` | 0.000245 |
| `MonthlyIncome` | 0.000234 |
| `NumberRealEstateLoansOrLines` | 0.000116 |
| `DebtRatio` | 0.000114 |
| `NumberOfTimes90DaysLate` | 0.000000 |

All CSI values are far below 0.10, indicating stable characteristic distributions.

---

## Calibration Monitoring

Calibration compares predicted PD against observed bad rate.

A well-calibrated model should produce average predicted PD values that are reasonably close to observed bad rates within each risk band.

If predicted PD is materially lower than observed bad rate, the model may underestimate risk.

If predicted PD is materially higher than observed bad rate, the model may overestimate risk.

Calibration monitoring is important because even a model with strong AUROC can become poorly calibrated over time.

---

## Key Conclusions

The final scorecard demonstrates:

- Strong discriminatory power
- Stable AUROC / GINI across train, validation and test samples
- No evidence of overfitting
- Strong rank ordering by score decile
- Stable score distribution across samples
- Stable model characteristics
- Clear and interpretable scorecard scaling
- A practical cut-off strategy based on approval rate and accepted bad rate

The model is suitable as a complete educational example of a traditional retail application scorecard development process.

---

## Requirements

The project uses the following main Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
optbinning
openpyxl
jupyter
kagglehub
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Download the Kaggle dataset.
2. Place raw files in `data/raw/`.
3. Run notebooks sequentially from business understanding to monitoring.
4. Review generated outputs in `data/outputs/`.

Suggested order:

```text
0_business_understanding.ipynb
0_data_extraction.ipynb
01_data_collection_and_quality.ipynb
02_sampling.ipynb
03_exploratory_data_analysis.ipynb
04_outlier_analysis.ipynb
05_variable_screening.ipynb
06_binning_methodology.ipynb
07_woe_transformation.ipynb
08_feature_selection.ipynb
09_model_performance.ipynb
10_scorecard_scaling.ipynb.ipynb
11_cutoff_determination.ipynb
12_score_run_and_fit.ipynb
13.validation_and_monitoring.ipynb
```

---
