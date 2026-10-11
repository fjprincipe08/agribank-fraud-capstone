# AI-Powered Fraud Detection and Risk Scoring System for Agribank Online Transactions

**AIM Artificial Intelligence and Machine Learning | Capstone Project**
Fritz Jerald V. Principe | Agribusiness Banking Corporation (Agribank)

> **Academic proof of concept on synthetic data.** This project uses the public PaySim dataset. It contains no Agribank customer or transaction data, and its results do not estimate performance at the Bank.

---

## Contents

1. [Overview](#overview)
2. [Problem framing](#problem-framing)
3. [Dataset](#dataset)
4. [Method](#method)
5. [Key results](#key-results)
6. [Risk scoring and explainability](#risk-scoring-and-explainability)
7. [Fairness review](#fairness-review)
8. [Limitations](#limitations)
9. [Repository structure](#repository-structure)
10. [How to reproduce](#how-to-reproduce)
11. [Deliverables](#deliverables)
12. [Rubric mapping](#rubric-mapping)
13. [Use of Generative AI](#use-of-generative-ai)
14. [References](#references)

---

## Overview

Agribank plans to add fund transfers to Agribank Online in 2027 through InstaPay (real-time) and PESONet (batch). More digital transfers mean more exposure to fraud, and manual review becomes harder to rely on as volumes grow. BSP Circular No. 1213 and Memorandum No. M-2024-029 (AFASA) also expect automated, real-time fraud monitoring and a Fraud Management System (FMS).

This project builds and evaluates a machine learning prototype that scores each transaction for fraud risk, converts the score to a 0 to 100 scale, and attaches the main reasons behind it. Fraud personnel decide which transactions to review; the model never blocks a transaction. The prototype is one possible component of a future FMS, not an FMS in itself.

**Main finding.** With every PaySim field, models reach a near-perfect PR-AUC of 1.000 because the simulator leaves obvious traces. Using only the details known when a transfer is submitted, the best model (XGBoost) reaches **PR-AUC 0.578**, about 45 times the rule-based benchmark. That second figure is the more honest starting point.

---

## Problem framing

| Element | Definition |
|---|---|
| Problem statement | How can machine learning help identify potentially fraudulent Agribank Online transactions, and assign a risk score that helps fraud personnel decide which transactions to review first? |
| Task type | Supervised binary classification with risk scoring |
| Target | `isFraud` (0 = legitimate, 1 = fraud) |
| Scoring point | Each transaction, before it completes, using only information available at that moment |
| Primary metric | PR-AUC (average precision) |
| Supporting metrics | Precision, recall, F1, ROC-AUC, false positive rate, Precision@k and Recall@k, Brier score |
| Business KPIs | Fraud detection rate, fraud value detected, alert volume, false alert volume, investigation efficiency, potential fraud loss exposure |

**Why not accuracy?** Fraud is 0.129% of records. A model that labels every transaction legitimate is 99.87% accurate and catches no fraud.

**Research questions**

| No. | Question |
|---|---|
| RQ1 | Which transaction and balance characteristics best separate fraudulent from legitimate transactions? |
| RQ2 | How much better does machine learning perform than the simple rule already in the dataset (`isFlaggedFraud`)? |
| RQ3 | How much do the results depend on balance patterns that may be simulator artifacts, or that Agribank could not see for transfers to other banks? |
| RQ4 | Which alert threshold gives a workable balance between fraud caught and review workload? |
| RQ5 | Do false alerts and missed fraud occur at noticeably different rates across groups of transactions? |

---

## Dataset

**Synthetic Financial Datasets for Fraud Detection (PaySim)**, Kaggle, file `PS_20174392719_1491204439457_log.csv` (Lopez-Rojas, Elmir and Axelsson, 2016).

| Item | Value |
|---|---|
| Transactions | 6,362,620 |
| Columns | 11 |
| Time span | 743 hourly steps (about 30 days) |
| Transaction types | PAYMENT, TRANSFER, CASH_IN, CASH_OUT, DEBIT |
| Fraud cases | 8,213 (0.129%, about 1 in 775) |
| Missing values | None |
| Duplicate rows | None |
| Amounts | Simulated currency units, not pesos |

**Data dictionary (summary)**

| Column | Type | Role | Known at scoring time? |
|---|---|---|---|
| `step` | Integer | Input (simulated hour) | Yes |
| `type` | Category | Input | Yes |
| `amount` | Numeric | Input | Yes |
| `nameOrig` | Identifier | History only | Yes |
| `oldbalanceOrg` | Numeric | Sender balance before | Yes |
| `newbalanceOrig` | Numeric | Sender balance after | Yes (old balance less amount) |
| `nameDest` | Identifier | History only | Yes |
| `oldbalanceDest` | Numeric | Receiver balance before | On-us transfers only |
| `newbalanceDest` | Numeric | Receiver balance after | No (known only after posting) |
| `isFraud` | Binary | Target | No (known after investigation) |
| `isFlaggedFraud` | Binary | Benchmark only | Not used as an input |

The full data dictionary, with ranges and units, is in the final report (PART V, Section C).

**Known simulator characteristics** confirmed during exploration:

- All 8,213 fraud cases are TRANSFER or CASH_OUT.
- 97.8% of fraud cases empty the sender account.
- 99.5% of fraudulent transfers are matched by a fraudulent cash-out of the same amount in the same hour.
- `isFlaggedFraud` flagged only 16 transactions in the whole dataset.

The CSV is not included in this repository. See [`data/README.md`](data/README.md).

---

## Method

### Preprocessing

1. Validated data types, labels and value ranges; checked for missing values and duplicates (none found).
2. Kept only TRANSFER and CASH_OUT, the two types with fraud (2,770,409 rows). This is documented as a business rule that would run before the model.
3. Kept extreme amounts: 338,078 rows (5.3%) above the IQR bound of 501,719.34 have a fraud rate of 1.14%, so removing them would remove fraud.
4. Split by time, not randomly:

| Set | Steps | Rows | Fraud cases | Fraud rate |
|---|---|---|---|---|
| Training | 1 to 322 | 1,938,484 | 3,633 | 0.19% |
| Validation | 323 to 376 | 415,628 | 560 | 0.13% |
| Test | 377 to 743 | 416,297 | 4,020 | 0.97% |

5. Class imbalance: class weights in every model, and legitimate training rows sampled down to 10% (197,118 rows, 1.84% fraud). Validation and test sets were never rebalanced.

**Leakage controls.** Statistics such as type medians and the high-value cut-off are learned on training rows only. Receiving-account history counts earlier hours only. `isFlaggedFraud` is a benchmark only. The receiving balance after the transfer is not available at scoring time.

### Feature engineering

| Family | Features |
|---|---|
| Amount | `log_amount`, `high_value`, `amt_vs_type_median` |
| Time | `hour`, `period_night`, `period_morning`, `period_afternoon` |
| Type | `type_TRANSFER` |
| Receiving-account history | `dest_prior_count`, `log_dest_prior_amount`, `dest_is_new` |
| Sender balances | `oldbalanceOrg`, `newbalanceOrig`, `amt_to_orig_bal` |
| Balance reconciliation | `errorBalanceOrig`, `errorBalanceDest`, `orig_emptied`, `dest_zero_after` |
| Receiving balances | `oldbalanceDest`, `newbalanceDest`, `amt_to_dest_bal` |

**Feature sets used in the stress tests**

| Set | Contents | Features | Test |
|---|---|---|---|
| A (full PaySim) | All features | 21 | E1, E4 |
| A without reconciliation | Set A less reconciliation features | 17 | E2 |
| B (interbank) | Set A less receiving balance features | 16 | E3 |
| C (no sender balances) | Set A less reconciliation and sender balance fields | 14 | E5 |
| **D (transaction details only)** | **Amount, type, time and receiving-account history** | **11** | **E6** |

Feature Set D is the realistic view: details known when a customer submits a transfer.

### Feature selection and dimensionality reduction

- **Feature selection:** mutual information, Random Forest importance and permutation importance on the validation set. A feature would be dropped only if weak on all three. None was dropped. All three ranked `orig_emptied`, a simulator artifact, first.
- **PCA:** 14 of 21 components were needed for 95% of variance. Logistic Regression validation PR-AUC fell from 0.9999 to 0.9891 with PCA, and components would hide readable reasons from reviewers. PCA was not adopted.

### Models

Rule-based benchmark (`isFlaggedFraud`), Logistic Regression, Decision Tree, Random Forest and XGBoost. Each was tuned by random search on validation PR-AUC. The test set was used once, for reporting.

---

## Key results

All results are on the test period (steps 377 to 743), before calibration unless stated.

### Realistic features (Feature Set D)

| Model | PR-AUC | 95% interval | ROC-AUC | Recall@150 | Precision@150 |
|---|---|---|---|---|---|
| Rule-based benchmark | 0.0129 | n/a | 0.5016 | n/a | n/a |
| Logistic Regression | 0.2694 | 0.256 to 0.284 | 0.8688 | 31.3% | 52.5% |
| Decision Tree | 0.5260 | 0.509 to 0.541 | 0.9328 | 43.4% | 72.7% |
| Random Forest | 0.5562 | 0.539 to 0.570 | 0.9416 | 43.9% | 73.5% |
| **XGBoost (selected)** | **0.5777** | **0.562 to 0.594** | **0.9422** | **45.3%** | **75.9%** |

Recall@150 and Precision@150 use the 150 highest scores per simulated day. Intervals come from 200 bootstrap resamples of the test set.

XGBoost was selected on validation PR-AUC (0.3696, against 0.3421 for the next model). Its interval overlaps Random Forest's, so the ranking between those two is uncertain. "About 45 times the rule" refers to ranking quality (PR-AUC), not to losses avoided.

**Selected XGBoost settings (Set D):** 400 trees, max depth 3, learning rate 0.05, min child weight 5, column sample 0.7, row sample 1.0.

### Stress tests (XGBoost, Set A settings)

| Test | What the model can see | Features | PR-AUC |
|---|---|---|---|
| E1 | All PaySim features | 21 | 1.0000 |
| E2 | No reconciliation features | 17 | 0.9994 |
| E3 | No receiving balances | 16 | 0.9998 |
| E4 | All features, random split | 21 | 0.9967 |
| E5 | No reconciliation or sender balances | 14 | 0.7945 |
| E6 | Transaction details only | 11 | 0.5568 |
| E6 tuned | Transaction details only, tuned on Set D | 11 | 0.5777 |

Each time one source of the simulator pattern was removed, the model found another. Performance dropped to a more realistic level only when all balance fields were removed.

---

## Risk scoring and explainability

**Risk score** = calibrated probability x 100. Isotonic regression was chosen by 5-fold Brier score inside the validation set.

**Prototype risk levels (E6, test period).** These ranges are illustrative, not approved banking thresholds.

| Level | Score | Transactions | Fraud cases | Fraud rate |
|---|---|---|---|---|
| Low | 0 to 29 | 414,530 | 2,505 | 0.60% |
| Medium | 30 to 69 | 1,054 | 834 | 79.13% |
| High | 70 to 100 | 713 | 681 | 95.51% |

**Capacity-based review (calibrated scores).** Daily capacities are assumptions, not Agribank staffing figures.

| Reviews per day | Recall | Precision | Fraud value caught | Total alerts |
|---|---|---|---|---|
| 50 | 18.1% | 91.0% | 38.3% | 800 |
| **150 (operating rule)** | **44.8%** | **75.0%** | **77.9%** | **2,400** |
| 300 | 62.8% | 52.8% | 90.4% | 4,782 |

**Calibration drift.** Calibration was fitted on a period with 0.13% fraud and applied to a test period with 0.97%. Scores understated risk, so the capacity rule, which ranks transactions within each day, is used as the main operating point.

**Explainability (SHAP).** The three most influential features in the realistic model are:

1. Time of day
2. Amount relative to the typical amount for its transaction type
3. Amount previously received by the receiving account

Each high-risk alert lists its top three reasons. A reason explains the score; it is not evidence that a customer committed fraud.

---

## Fairness review

PaySim has no customer attributes such as age, sex, location or income, so **no demographic fairness claim can be made**. The review compares error rates across transaction groups (transaction type, amount range, sender balance range and time of day) under the 150-per-day rule.

- Night-time transactions had the highest false positive rate, at **37.4 times** the overall rate.
- Time of day is the only grouping flagged beyond its fraud-rate gap.

**Mitigation test: retrain without the four time-of-day features**

| Measure | With time features | Without time features |
|---|---|---|
| Test PR-AUC (both before calibration) | 0.5777 | 0.4190 |
| FPR gap across time-of-day groups | 0.054 | 0.003 |
| Detection-rate gap across time-of-day groups | 0.628 | 0.057 |

Removing time features narrowed the disparity but reduced detection. This is a trade-off, not a resolution. Reweighting low-detection groups was not tested.

---

## Limitations

- **Simulated data.** Mobile-money activity may differ greatly from InstaPay and PESONet transfers.
- **Overstated performance.** Even the realistic result may be optimistic, since real data is less consistent.
- **Short period.** About one month of data; changing fraud patterns could not be tested.
- **Small number of fraud cases.** The validation set had 560 fraud cases.
- **Leakage risk.** Post-transaction balances and the simulator's own flag can reveal the label.
- **Calibration drift.** Scores need regular recalibration.
- **Time-of-day pattern.** May be a simulator artifact.
- **Missing production data.** A real FMS would need device, login, location, velocity and blacklist data that PaySim does not have.

---

## Repository structure

```
agribank-fraud-capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── fraud_capstone_pipeline.py      Full pipeline, run top to bottom
├── notebooks/
│   └── fraud_capstone_pipeline.ipynb   Notebook version of the pipeline
├── data/
│   ├── README.md                       Where to download PaySim
│   └── raw/                            Place the PaySim CSV here (not committed)
├── models/
│   ├── run_config.json                 Settings, feature sets, selected models, versions
│   └── *.joblib                        Saved models and calibrators
├── reports/
│   ├── FJVP-AIML-Capstone_Technical_Report_Final.pdf
│   ├── figures/                        Charts produced by the pipeline
│   └── tables/                         Result tables, report_numbers.json, claim checks
└── presentations/
    ├── technical/                      Jupyter slide deck (.ipynb and .html)
    └── business/                       PowerPoint deck for executives
```

---

## How to reproduce

**Requirements:** Python 3.10 (shap 0.49.1 is the last version supporting it).

```bash
# 1. Clone the repository
git clone https://github.com/fjprincipe08/agribank-fraud-capstone.git
cd agribank-fraud-capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download PaySim from Kaggle into data/raw/ (see data/README.md)

# 4. Run the pipeline from the repository root
python src/fraud_capstone_pipeline.py
```

**Outputs**

| Path | Contents |
|---|---|
| `models/` | Saved models, calibrators and `run_config.json` |
| `reports/figures/` | All charts |
| `reports/tables/` | All result tables |
| `reports/tables/report_numbers.json` | Every number quoted in the report |
| `reports/tables/17_report_claim_checks.csv` | Automatic check of each claim the report makes in words |

**Reproducibility settings**

| Item | Value |
|---|---|
| Random seed | 42 (sampling, training and bootstrap) |
| Data split | Time-based: steps 1 to 322, 323 to 376, 377 to 743 |
| Model selection | Validation PR-AUC only; models within 0.002 treated as tied |
| Calibration and thresholds | Chosen on validation only; test used once |
| Versions | pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, xgboost 3.2.0, shap 0.49.1, matplotlib 3.10.8 |

For a quick dry run, set `"nrows": 500_000` in the `CONFIG` dictionary at the top of the script. Set it back to `None` for the full run.

---

## Deliverables

| Deliverable | Location |
|---|---|
| Final technical report | [`reports/`](reports/) |
| Reproducible code | [`src/fraud_capstone_pipeline.py`](src/fraud_capstone_pipeline.py) |
| Saved models and configuration | [`models/`](models/) |
| Technical presentation (Jupyter slides) | [`presentations/technical/`](presentations/technical/) |
| Business presentation (PowerPoint) | [`presentations/business/`](presentations/business/) |

**Viewing the technical presentation:** download `FJVP_AIML_Capstone_Technical_Deck.html` and open it in a browser. It works offline. Press `S` for speaker notes.

---

## Rubric mapping

Before submission, the final report was reviewed against the AIM capstone rubric, step by step, to confirm that every requirement is addressed and can be traced to the code. The table below maps each rubric step to the report, the pipeline script and this README.

| Rubric step | What it covers | Final report | Pipeline (`src/fraud_capstone_pipeline.py`) | README |
|---|---|---|---|---|
| Step 1 | Problem framing: business problem, task type and why, success metrics, business KPIs | PART III and PART IV | Section 1 | [Problem framing](#problem-framing) |
| Step 2 | Data collection and understanding: source, data dictionary with ranges and units, quality checks, known simulator characteristics | PART V | Section 2 | [Dataset](#dataset) |
| Step 3 | Exploratory analysis, preprocessing, outlier handling, feature engineering, class imbalance, feature selection and PCA | PART VI and PART VII | Sections 3 to 6 | [Method](#method) |
| Step 4 | Model training and tuning, model comparison, stress tests, calibration, risk score and thresholds | PART VIII and PART IX | Sections 7 to 11 | [Key results](#key-results), [Risk scoring and explainability](#risk-scoring-and-explainability) |
| Step 5 | Explainability, bias and fairness review with a tested mitigation, limitations | PART X, PART XI and PART XII | Sections 12 to 14 | [Risk scoring and explainability](#risk-scoring-and-explainability), [Fairness review](#fairness-review), [Limitations](#limitations) |
| Step 6 | Technical presentation and business presentation | PART XIII and PART XIV | Section 15 | [Deliverables](#deliverables) |
| Step 7 | GitHub repository, final report and reproducible code | Whole report | Sections 16 and 17 | [Repository structure](#repository-structure), [How to reproduce](#how-to-reproduce) |

**Traceability between the report and the code.** Every number quoted in the report is saved by the pipeline to `reports/tables/report_numbers.json`. Every claim the report makes in words (for example, that removing the reconciliation features barely changes PR-AUC) is checked automatically in Section 17 and recorded in `reports/tables/17_report_claim_checks.csv`.

---

## Use of Generative AI

Generative AI tool was used during this project to assist with code review and some debugging, cleanup of the pipeline script, and drafting and editing of documentation, as well as mapping the output against the instructions provided.

---

## References

- Bangko Sentral ng Pilipinas. (2024). *Memorandum No. M-2024-029.* Guidelines on risk management systems and controls under the Anti-Financial Account Scamming Act (AFASA), Republic Act No. 12010.
- Bangko Sentral ng Pilipinas. (2025). *Circular No. 1213.* Amendments to the regulations relating to electronic products and services and fraud management requirements.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*
- Kaggle. *Synthetic Financial Datasets for Fraud Detection (PaySim).*
- Lopez-Rojas, E. A., Elmir, A., and Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *Proceedings of the 28th European Modeling and Simulation Symposium (EMSS).*
- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30.*
- Saito, T., and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3).

---

*Academic capstone project. Not a production system and not an approved Agribank initiative.*
