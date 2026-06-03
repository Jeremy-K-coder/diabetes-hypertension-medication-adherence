## **PREDICTING MEDICATION NON-ADHERENCE IN DIABETES AND HYPERTENSION PATIENTS USING MACHINE LEARNING**
### Individual Project: Classical Machine Learning for Health Data
#### NAME: **`Kirunda Jeremy Menya`**
#### REG NO.: **`2025/HD07/25995U`**
#### STUDENT NO.: **`2500725995`**

*Makerere University | June 2026*

![Adherence Overview](figures/adherence_overview.png)
*Medication adherence rate breakdown by condition and insurance tier across the study cohort (Zimbabwe, 2022).*

---

## Contents
1. [Project Overview](#project-overview)
2. [Motivation & Significance](#motivation--significance)
3. [Research Gap & Originality](#research-gap--originality)
4. [Clinical Background](#clinical-background)
5. [Dataset](#dataset)
6. [Repository Structure](#repository-structure)
7. [Methodology](#methodology)
8. [Experimental Design](#experimental-design)
9. [Results](#results)
10. [Explainability (SHAP)](#explainability-shap)
11. [Setup and Usage](#setup-and-usage)
12. [Ethical Considerations](#ethical-considerations)
13. [References](#references)

---

## Project Overview

Non-communicable diseases (NCDs), chiefly Type 2 Diabetes and Hypertension, are the fastest-growing cause of death across Sub-Saharan Africa, now accounting for over 37% of adult mortality in the region. Yet medication adherence rates among NCD patients in Zimbabwe and similar settings remain below 50%, far lower than in high-income countries. When patients do not take their medications consistently, disease progresses silently, complications compound, hospitalisations increase, and healthcare systems that are already under-resourced bear an even greater burden.

This project builds and evaluates a classical machine learning pipeline to **predict medication non-adherence** among diabetic and hypertensive patients in Zimbabwe, using pharmacy refill records and patient-level insurance data from **Cimas Medical Aid Society**, one of Zimbabwe's largest health insurance providers. The dataset is real, prospectively collected, and sourced from a published academic study (Kanyongo et al., 2024), making the findings directly relevant to clinical decision-making in a genuine Sub-Saharan African health system context.

Beyond standard prediction, this project introduces two original analytical contributions:

1. **Feature group comparison experiment:** clinical consumption features (units dispensed, refill gaps) and socioeconomic proxy features (insurance tier, cost burden ratio) are evaluated separately and in combination, asking which feature category drives predictive performance. This question carries direct policy implications for community health worker targeting programmes.

2. **Clinical cost-sensitive evaluation:** a 2×2 misclassification cost matrix that penalises false negatives (predicting adherent when a patient is not) more heavily than false positives, reflecting the real consequence of failing to identify a non-adherent patient in a resource-constrained clinic.

**Core research questions this project addresses:**
- Can pharmacy refill and insurance data predict medication non-adherence with clinically meaningful accuracy in a Zimbabwean NCD population?
- Do features serving as socioeconomic indicators (insurance tier, cost burden) predict adherence as well as clinical consumption features (refill frequency, units dispensed)?
- Which features contribute most to adherence predictions, and what do those contributions imply for community-level intervention design?
- Which classical ML algorithm achieves the best balance between predictive performance and clinical safety under a cost-sensitive evaluation framework?

---

## Motivation & Significance of this project

### The NCD Crisis in Sub-Saharan Africa (SSA)

The epidemiological transition underway in Sub-Saharan Africa has been described as a double burden: communicable diseases have not yet been fully contained, while NCDs are now rising rapidly. The International Diabetes Federation estimates that by 2045, Africa will have the fastest-growing diabetes epidemic globally, with a 129% increase in prevalence projected. Hypertension already affects an estimated 30% of adults in Sub-Saharan Africa, yet awareness and treatment rates remain among the lowest in the world.

In Zimbabwe, this burden is compounded by structural barriers to care: shortages of specialist physicians, uneven access to medications, a fragmented insurance landscape, and high out-of-pocket costs. Against this backdrop, medication adherence is not a behavioural nuance. It is the primary determinant of whether treatment translates into health outcomes at all.

### Why Adherence Prediction Matters

The clinical and economic consequences of non-adherence are severe and well-documented:

- **Clinical:** uncontrolled diabetes leads to retinopathy (preventable blindness), nephropathy, neuropathy, and cardiovascular events. Uncontrolled hypertension causes stroke, heart failure, and chronic kidney disease. Both conditions are significantly cheaper and more effective to manage when adherence is maintained.
- **Economic:** studies in Sub-Saharan Africa estimate that hospitalisation costs attributable to non-adherence are three to five times higher than the cost of the medications themselves.
- **Systemic:** in resource-limited settings, avoidable hospitalisations consume capacity that is already critically scarce.

Identifying non-adherent patients early, before complications manifest, opens a window for targeted pharmacist counselling, community health worker visits, medication synchronisation programmes, and simplified regimen adjustments. **This is the intervention opportunity that a trained ML model enables.**

### Why Zimbabwe, Why This Dataset

Most adherence prediction research uses self-reported survey instruments (MMAS-8, GMAS), which are subject to social desirability bias and recall error. The Kanyongo et al. (2024) dataset is based on objective pharmacy refill records from a formal medical aid system, not patient self-report. This makes it one of the few SSA-origin datasets capable of generating *unbiased* adherence labels at scale.

The dataset originates from Cimas Medical Aid Society, a well-established and regulated Zimbabwean insurer with standardised electronic claims data, making it directly representative of the formal health sector in Zimbabwe and broadly relevant to comparable insurance-based health systems across Southern and Eastern Africa, including South Africa, Kenya, Zambia, and Malawi.

---

## Research Gap & Originality

The Kanyongo (2024) PhD thesis that produced this dataset applied seven ML classifiers to the final dataset and used RF feature importance and LIME for interpretation. This project **does not repeat that work.** It introduces three original contributions not present in any prior publication using this dataset:

### 1. Feature Group Experiment (Primary Contribution)
The dataset features are partitioned into two clinically meaningful groups:
- **Group A: Socioeconomic Proxies** — `insurance_tier`, `cost_burden_ratio`, `age_group`, `gender`, `wellness_programme_member`
- **Group B: Clinical Consumption** — `annual_units`, `refill_gap_days`, `refill_regularity`, `units_per_refill`, `annual_claim_amount`

All five classifiers are first trained on the combined feature set (Group C) to establish a performance ranking and identify the best model. The feature group experiment then runs on the **best tuned model only**, comparing Group A, Group B, and Group C performance. This sequencing ensures the group comparison reflects the strongest possible model rather than an untuned baseline. The research question — *do socioeconomic features predict adherence as well as clinical consumption features?* — has direct implications for whether community health workers (who may not have access to pharmacy dispensing records) can use patient-level socioeconomic information to identify non-adherent individuals.

### 2. Engineered Features Not Present in the Raw Dataset
Several features used in this project are computed during feature engineering and do not exist in the Mendeley dataset as released:
- `cost_burden_ratio = AnnualClaimAmount / AnnualContribution` (financial stress proxy)
- `refill_gap_days` (mean days between consecutive pharmacy visits, computed from service date records)
- `refill_regularity` (standard deviation of refill gaps; irregular refills are a behavioural adherence signal)
- `units_per_refill = annual_units / number_of_refills` (medication intensity proxy)
- `is_comorbid` (1 if patient appears in both diabetic and hypertension patient files)
- `insurance_tier` (ordinal encoding of OptionName: Basic < Standard < Premier)

### 3. Cost-Sensitive Evaluation
A clinical misclassification cost matrix is applied at evaluation time. Predicting a non-adherent patient as adherent (false negative) receives a higher cost than the reverse, because it directly results in a missed intervention opportunity. This reframes model comparison beyond raw accuracy and F1, producing a metric that clinicians and health programme managers would find directly actionable.

---

## Clinical Background

### Medication Adherence

The World Health Organisation defines medication adherence as "the degree to which a person's behaviour, taking medication, following a diet, and executing lifestyle changes, corresponds with agreed recommendations from a health care provider." In this project, adherence is operationalised using the **Medication Possession Ratio (MPR)**: a patient is adherent if they filled at least **75% of their expected 12-month medication supply** during 2022 (i.e. ≥9 refills out of 12 projected). This threshold is standard in pharmacy claims-based adherence research.

### Type 2 Diabetes Mellitus

A metabolic disorder characterised by insulin resistance and progressive beta-cell failure, leading to chronic hyperglycaemia. Management requires lifelong pharmacotherapy (oral hypoglycaemics, insulin), dietary modification, and regular monitoring. Non-adherence to antidiabetic medications accelerates microvascular complications (retinopathy, nephropathy, neuropathy) and macrovascular events (myocardial infarction, stroke). In Sub-Saharan Africa, late-stage diagnosis and fragmented follow-up care create compounding adherence challenges.

### Hypertension

A chronic condition defined by persistently elevated blood pressure (≥140/90 mmHg), which is the leading modifiable risk factor for cardiovascular disease globally. Antihypertensive medications must typically be taken daily for life. Because hypertension is largely asymptomatic, patients who feel well frequently discontinue medications (a pattern sometimes called the "I feel fine" adherence failure). In Zimbabwe, less than 30% of hypertensive patients achieve target blood pressure control, largely due to adherence failure.

### Comorbidity

Many patients in this dataset carry both diabetes and hypertension simultaneously, a common and clinically significant combination. Comorbid patients face greater medication burden (polypharmacy), higher treatment complexity, and elevated non-adherence risk. The `is_comorbid` feature engineered in this project specifically captures this group for analysis.

---

## Dataset

**Dataset for analysing medication adherence among diabetes and hypertension patients: Patient-level and medication refill data**
Kanyongo et al. (2024): Pharmacy refill and patient-level claims data from **Cimas Medical Aid Society**, Harare, Zimbabwe.

| Property | Detail |
|:---|:---|
| Source | [Mendeley Data](https://data.mendeley.com/datasets/zkp7sbbx64/2) |
| DOI | 10.17632/zkp7sbbx64.2 |
| Licence | CC0 1.0 (fully open, no restrictions) |
| Collection period | January 1 to December 31, 2022 |
| Total patients (final dataset) | ~8,141 unique patients |
| Conditions | Type 2 Diabetes, Hypertension |
| Origin | Cimas Medical Aid Society, Zimbabwe |
| Adherence definition | MPR ≥ 75% = Adherent; MPR < 75% = Non-Adherent |
| Label type | Binary |

**Files in the Mendeley repository:**

| File | Description | Used in this project |
|:---|:---|:---:|
| `Data before wrangling (Diabetes).xlsx` | Raw insurance transaction records, diabetes ICD-10 codes. Multiple rows per patient, no adherence label. | ❌ No (documented only) |
| `Data before wrangling (HTN).xlsx` | Raw insurance transaction records, hypertension ICD-10 codes. Multiple rows per patient, no adherence label. | ❌ No (documented only) |
| `Final Dataset.xlsx` | Cleaned, one-row-per-patient dataset with engineered label and aggregated features. | ✅ Yes |

**Why the final dataset only?**
The raw wrangling files require nine iterative data preparation steps (ICD-10 filtering, deduplication, date parsing, aggregation, label computation) documented in Kanyongo's PhD thesis. These steps are described in `notebooks/01_EDA.ipynb` to demonstrate understanding of data provenance, but all modelling uses the final dataset directly to avoid replicating the thesis methodology unnecessarily.

**Class distribution:**

![Adherence Distribution](figures/adherence_distribution.png)
*Adherence vs. non-adherence breakdown across diabetic and hypertensive patient groups.*

| Label | Class | Approximate Count |
|:---|:---|:---|
| 1 | Adherent (MPR ≥ 75%) | ~5,700 |
| 0 | Non-Adherent (MPR < 75%) | ~2,441 |

*Exact counts populated after EDA (01_EDA.ipynb).*

---

## Repository Structure

```
diabetes-hypertension-medication-adherence/
│
├── data/
│   └── raw/                                  # Final Dataset.xlsx — not tracked by git
│
├── notebooks/
│   ├── 01_EDA.ipynb                          # Data loading, provenance, class balance, feature distributions
│   ├── 02_Feature_Engineering.ipynb          # Feature construction, group definition, correlation analysis
│   ├── 03_Baseline_Models.ipynb              # All 5 classifiers on Group C (combined features); model ranking
│   ├── 04_Optimization_Calibration.ipynb     # Hyperparameter tuning and calibration for top 2 models
│   └── 05_Explainability_CostAnalysis.ipynb  # Feature group experiment, SHAP analysis, clinical cost evaluation
│
├── src/
│   ├── features.py                           # Feature engineering functions (ratios, gaps, regularity)
│   ├── pipeline.py                           # sklearn Pipeline builder (scaler + encoder + model)
│   ├── evaluate.py                           # Metrics: accuracy, F1, AUC, MCC, clinical cost score
│   ├── cost_matrix.py                        # 2×2 clinical cost matrix and weighted scoring
│   └── utils.py                              # Reproducibility helpers, plotting utilities
│
├── models/                                   # Saved model pickles — not tracked by git
│
├── reports/
│   └── final_report.pdf                      # [View Report](YOUR_DRIVE_LINK_HERE)
│
├── figures/
│   ├── adherence_overview.png                # Header figure: adherence by condition and tier
│   ├── adherence_distribution.png            # Class balance plot
│   ├── feature_importance_shap.png           # SHAP beeswarm plot (best tuned model)
│   ├── feature_group_comparison.png          # Group A vs B vs C on best tuned model
│   ├── confusion_matrix_best.png             # Confusion matrix of best model on test set
│   ├── roc_auc_comparison.png                # ROC curves across all five classifiers
│   ├── learning_curves/
│   │   └── best_model_learning_curve.png     # Learning curve of best tuned model
│   ├── calibration/
│   │   └── calibration_comparison.png        # Reliability diagram across classifiers
│   └── cost_analysis/
│       └── clinical_cost_comparison.png      # Cost-weighted error comparison across models
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Methodology

### Feature Engineering

Raw features in the final dataset are supplemented with engineered features before any modelling:

| Feature | Formula / Source | Clinical Rationale |
|:---|:---|:---|
| `cost_burden_ratio` | `AnnualClaimAmount / AnnualContribution` | Financial stress proxy; high burden predicts lower adherence |
| `refill_gap_days` | Mean days between consecutive pharmacy visits | Long gaps signal lapses in medication collection behaviour |
| `refill_regularity` | Std. dev. of refill gaps | Irregular refills indicate inconsistent adherence behaviour |
| `units_per_refill` | `annual_units / n_refills` | Medication intensity and dose adjustment proxy |
| `is_comorbid` | 1 if patient in both diabetic and HTN files | Comorbid patients face higher pill burden and non-adherence risk |
| `insurance_tier` | Ordinal: Basic=0, Standard=1, Premier=2 | Socioeconomic status proxy from insurance OptionName |
| `age_group` | Binned: 18–34, 35–49, 50–64, 65+ | Age is a documented adherence predictor in SSA literature |

### Feature Groups (The Core Experiment)

| Group | Features Included | Clinical Question |
|:---|:---|:---|
| **A: Socioeconomic** | `insurance_tier`, `cost_burden_ratio`, `age_group`, `gender`, `wellness_programme_member` | Can CHWs use patient-level data alone to identify non-adherent patients? |
| **B: Clinical Consumption** | `annual_units`, `refill_gap_days`, `refill_regularity`, `units_per_refill`, `annual_claim_amount` | How much predictive signal is in pharmacy dispensing behaviour alone? |
| **C: Combined** | All features from A and B plus `is_comorbid`, `diagnosis`, `complication_flag` | Full model for deployed clinical decision support |

### Preprocessing Pipeline

All preprocessing is implemented inside `sklearn.Pipeline` objects to guarantee zero data leakage between train and test sets:
- `StandardScaler` on all continuous features
- `OneHotEncoder` on nominal categoricals (`gender`, `diagnosis`)
- `OrdinalEncoder` on ordered categoricals (`insurance_tier`, `age_group`)
- **SMOTE** applied to the training fold only (inside cross-validation) to handle class imbalance without contaminating the test set

### Data Splitting

Stratified 70/15/15 train/validation/test split (seed=42). Stratification preserves the adherence class ratio in each subset.

### Classifiers

Five classical ML classifiers are trained and evaluated:

| Classifier | Rationale |
|:---|:---|
| Logistic Regression | Interpretable baseline; log-odds directly map to clinical risk factors |
| Decision Tree | Fully interpretable rule-based model; relevant for clinical guideline translation |
| Random Forest | Ensemble robustness with built-in feature importance and strong tabular performance |
| Gradient Boosting (XGBoost) | State-of-the-art for tabular data; handles missing values natively |
| Support Vector Machine (RBF) | Effective in high-dimensional patient feature spaces |

### Hyperparameter Optimisation

`RandomizedSearchCV` with 5-fold stratified cross-validation on the best two classifiers from baseline evaluation. Optimised on Macro F1 rather than accuracy to account for class imbalance.

---

## Experimental Design

The experiment runs in two sequential phases, with the feature group comparison deliberately placed after tuning rather than before it. Running the group experiment on untuned models risks drawing conclusions that do not hold under the best possible model capacity.

**Phase 1: Baseline (Notebook 03)**
All five classifiers are trained on Group C (the full combined feature set) to establish a performance ranking. This phase identifies the top two models and confirms which algorithms are worth tuning.

**Phase 2: Optimisation and Calibration (Notebook 04)**
The top two classifiers from Phase 1 undergo `RandomizedSearchCV` hyperparameter tuning followed by calibration curve analysis, assessing whether predicted probabilities are reliable enough for clinical risk stratification.

**Phase 3: Feature Group Experiment, Explainability, and Clinical Cost (Notebook 05)**
The best tuned model from Phase 2 is then evaluated across Group A, Group B, and Group C. This is where the core research question is answered. SHAP analysis follows on the Group C model, and the clinical cost framework is applied across all Phase 1 and Phase 2 results.

All runs share: identical train/validation/test splits, the same SMOTE random seed, the same evaluation metrics, and the same cross-validation folds.

| Metric | Why included |
|:---|:---|
| Accuracy | Overall correctness baseline |
| Macro F1 | Equal weight per class; penalises minority class failure |
| Weighted F1 | Accounts for class frequency in aggregation |
| ROC-AUC | Discrimination power independent of threshold |
| MCC (Matthews Correlation Coefficient) | Most robust single metric for imbalanced binary classification |
| Precision / Recall | Recall of non-adherents is the safety-critical metric in clinical deployment |
| Clinical Cost Score | Sum of (confusion count × cost weight) across FP and FN cells |

---

## Results

![Feature Group Comparison](figures/feature_group_comparison.png)
*Macro F1 across feature groups A, B, and C on the best tuned model. Populated after running Notebook 05.*

| Classifier | Feature Group | Accuracy | Macro F1 | AUC | MCC | Clinical Cost |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | C (Combined) | — | — | — | — | — |
| XGBoost | C (Combined) | — | — | — | — | — |
| Logistic Regression | C (Combined) | — | — | — | — | — |
| SVM (RBF) | C (Combined) | — | — | — | — | — |
| Decision Tree | C (Combined) | — | — | — | — | — |

*Full results table populated after running `03_Baseline_Models.ipynb` and `04_Optimization_Calibration.ipynb`.*

![Confusion Matrix](figures/confusion_matrix_best.png)
*Confusion matrix of the best-performing model on the held-out test set.*

![ROC Curves](figures/roc_auc_comparison.png)
*ROC-AUC curves across all five classifiers on the combined feature group.*

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is applied to the best tuned model to provide both global and local interpretability.

![SHAP Beeswarm](figures/feature_importance_shap.png)
*SHAP beeswarm plot showing the contribution of each feature to adherence predictions. Each point is a patient; colour indicates feature value (red = high, blue = low).*

Three analytical questions guide SHAP interpretation:

- **Group dominance:** Do features from Group A (socioeconomic) or Group B (clinical consumption) carry higher mean absolute SHAP magnitude in the combined model? This answers the core research question about community health worker targeting.
- **Direction of effects:** Does higher `cost_burden_ratio` increase or decrease predicted adherence probability? Does longer `refill_gap_days` predict non-adherence as expected?
- **Comorbidity effect:** What is the SHAP contribution of `is_comorbid` relative to single-condition patients? Does dual diagnosis predict worse adherence?

SHAP dependence plots are generated for the three features with the highest mean absolute SHAP value to visualise interaction effects.

---

## Setup and Usage

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Jeremy-K-coder/diabetes-hypertension-medication-adherence.git
cd diabetes-hypertension-medication-adherence
pip install -r requirements.txt
```

Place `final_dataset.xlsx` (downloaded from Mendeley Data) under `data/raw/`:

```
data/
└── raw/
    └── final_dataset.xlsx
```

**Running on Google Colab:**
```python
!git clone https://github.com/Jeremy-K-coder/diabetes-hypertension-medication-adherence.git
%cd diabetes-hypertension-medication-adherence
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy(
    '/content/drive/MyDrive/diabetes-hypertension-medication-adherence/final_dataset.xlsx',
    'data/raw/final_dataset.xlsx'
)
```

Notebooks are designed to run in order (01 through 05). Each notebook saves its key outputs (cleaned dataframes, feature matrices, trained models, figures) so subsequent notebooks can load them without re-running earlier steps.

| Notebook | Inputs | Key Outputs |
|:---|:---|:---|
| `01_EDA.ipynb` | `data/raw/Final Dataset.xlsx` | `data/processed/cleaned.pkl`, EDA figures |
| `02_Feature_Engineering.ipynb` | `cleaned.pkl` | `data/processed/features_A.pkl`, `features_B.pkl`, `features_C.pkl` |
| `03_Baseline_Models.ipynb` | `features_C.pkl` | Model ranking, ROC curves, confusion matrices |
| `04_Optimization_Calibration.ipynb` | `features_C.pkl` + baseline ranking | Tuned model pickles, learning curves, calibration plots |
| `05_Explainability_CostAnalysis.ipynb` | All feature group pickles + best model pickle | Feature group comparison, SHAP plots, clinical cost figures |

---

## Submission Links

| Deliverable | Link |
|:---|:---|
| 📄 Final Report (PDF) | [Google Drive](Link when report uploaded) |
| 💻 GitHub Repository | [medication-adherence-ml](https://github.com/Jeremy-K-coder/diabetes-hypertension-medication-adherence) |
| 🎞️ Presentation Slides | [Google Drive](Link to slides when uploaded) |

---

## Ethical Considerations

- **Patient privacy:** The dataset is sourced from a published, peer-reviewed repository under CC0 1.0 licence. All patient records were de-identified by the original data collectors at Cimas Medical Aid Society before deposit. No patient-identifiable information (name, ID number, address) is present in any file used in this project.
- **Fairness and generalisability:** The dataset reflects a single insurer's member population in Harare, Zimbabwe. Cimas members are disproportionately formally employed, urban, and relatively higher-income compared to the general Zimbabwean population. Predictions from this model may not generalise to informal sector workers, rural populations, or patients in public-sector facilities where medication supply chains differ substantially.
- **Deployment caution:** This project is an academic research prototype. It is not validated for clinical deployment. Any real-world application of adherence prediction in a clinical or programme setting would require prospective validation on a representative population, stakeholder engagement with patients and clinicians, and transparent communication about model limitations and error rates.
- **Bias in the adherence label:** The 75% MPR threshold, while standard in pharmacy research, equates a patient who missed one month of medication with one who missed four. Both are labelled non-adherent, but their clinical risk profiles differ substantially. This is a known limitation of claims-based adherence labelling and is discussed explicitly in the Limitations section of the final report.
- **Intervention intent:** The purpose of adherence prediction in this context is to *support targeted, helpful intervention* and not to penalise patients or restrict their access to care. Any deployment framing must make this intent explicit.

---

## References

- Kanyongo, W., Moyo, T., Ezugwu, A. E-S., & Fonou-Dombeu, J. V. (2024). Dataset for analysing medication adherence among diabetes and hypertension patients: Patient-level and medication refill data. *Mendeley Data*, V2. https://doi.org/10.17632/zkp7sbbx64.2
- Kanyongo, W. (2024). *Medication adherence classification for non-communicable disease patients through machine learning approaches* (Doctoral thesis). University of KwaZulu-Natal. https://hdl.handle.net/10413/23561
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*. https://arxiv.org/abs/1603.02754
- World Health Organisation. (2019). *World Report on Vision*. WHO, Geneva.
- International Diabetes Federation. (2023). *IDF Diabetes Atlas* (11th ed.). https://www.diabetesatlas.org
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
