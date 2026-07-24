**English** | [繁體中文](README.zh-TW.md)

---

# Semicon Yield Prediction

> Semiconductor Yield Prediction Model — Uses SECOM sensor data to predict wafer lot Pass / Fail,
> and identifies key influencing sensors with SHAP.

> 🔗 Sister project: [semicon-yield-dashboard](https://github.com/tzuhua0308/semicon-yield-dashboard)
> (This repo directly connects to that project's PostgreSQL, sharing the same SECOM data)

---

## 🎯 Project Goal

Predict "**will this wafer lot Fail?**" from 590 process sensor readings. Real-world fab equivalents:

- Flag likely-Fail lots within **5 minutes** of leaving the equipment → send back to metrology station early
- Identify sensors that **most influence yield** (via SHAP) → help engineers root-cause investigate

---

## 📦 Data Source

| Item | Details |
|---|---|
| Dataset | UCI SECOM Dataset (shared via sister project's PostgreSQL) |
| Samples | 1,567 wafer lots |
| Features | 590 sensor readings |
| Label | Pass (good) = 0, Fail (bad) = 1 |
| **Class Ratio** | **14.1 : 1** (severely imbalanced) ⚠️ |
| Missing Values | 4.54% NaN |

---

## 🗺️ Roadmap

- [x] **Day 1** (6/29): Baseline XGBoost — F1 = 0.00, AUC = 0.74 (demonstrates the "Majority Class Predictor" trap)
- [x] **Day 2** (7/1): Feature cleaning + mutual_info selection + OOF threshold tuning → **F1 = 0.17 (test) / 0.26 (5-fold OOF)**
- [ ] **Day 3**: 4-model showdown (RF / XGBoost / LightGBM / CatBoost) + SMOTE
- [ ] **Day 4**: Optuna Bayesian tuning + SHAP interpretability (find Top 10 yield-impacting sensors)
- [ ] **Day 5**: Streamlit Demo (input sensor values → real-time Pass/Fail prediction)
- [ ] **Day 6-7**: README polish, screenshots, GitHub push

---

## 📊 Day 1 Baseline Results

```
Confusion Matrix:
              Predict Pass   Predict Fail
Actual Pass       291              2
Actual Fail        21              0    ← All 21 Fails missed!

F1 (Fail) : 0.0000   ❌ Failed to catch any bad lot
ROC AUC   : 0.7395   ✅ Model has signal — threshold is wrong
Accuracy  : 93%      ⚠️ Illusory (predicting all Pass already gives 93%)
```

### Lessons Learned (Intentionally Highlighted)

**This baseline looks like a "failure," but reveals 3 important concepts:**

1. **Accuracy is meaningless for imbalanced classification** — "Predict all majority class" gives 93% while doing nothing.
2. **AUC 0.74 indicates the model has signal** — the default threshold = 0.5 is just too conservative.
3. **This is the classic "Yield Excursion early warning" problem in industry** — Fail samples are naturally rare, a textbook imbalanced learning scenario.

Day 2 uses **Feature Selection + Threshold Tuning** to lift F1 from 0.

---

## 📈 Day 2 Improved Results

Run: `python src/02_improved.py` (same train/test split, same XGBoost params — only add cleaning and threshold tuning).

```
── Baseline (Day 1) ────────────────────
Confusion Matrix:
              Predict Pass   Predict Fail
Actual Pass       291              2
Actual Fail        21              0    ← Caught 0
F1 = 0.0000  AUC = 0.7395  threshold = 0.5

── Improved (Day 2) ────────────────────
Confusion Matrix:
              Predict Pass   Predict Fail
Actual Pass       282             11
Actual Fail        18              3    ← Caught 3 (baseline caught 0)
F1 (test)     = 0.1714
F1 (5-fold OOF) = 0.2642   ← more stable estimate
AUC          = 0.7549  threshold = 0.120
```

### Four Techniques Applied

| # | Step | Effect |
|---|---|---|
| 1 | Drop sensors with NaN ≥ 50% | 590 → 566 columns |
| 2 | Median imputation (fit only on train) | Prevents NaN from distorting mutual_info |
| 3 | VarianceThreshold to drop near-constant columns | 566 → 440 columns |
| 4 | `mutual_info_classif` Top-50 | 440 → 50 columns, filters noise |
| 5 | `scale_pos_weight = 14` | Inherited from Day 1, imbalance weighting |
| 6 | 5-fold StratifiedKFold OOF probabilities | Avoids overfitting threshold selection to single split |
| 7 | Sweep threshold for best F1 | **0.50 → 0.12 (the real hero)** |

### Why is Day 1 baseline F1 = 0?

Not because the model is bad (AUC 0.74 shows ranking ability exists) — it's because **threshold = 0.5 is unreasonable for this 14:1 imbalance**. XGBoost's `predict_proba` generally gives low probabilities on the minority class (even with `scale_pos_weight`); most Fails fall below 0.5 and get missed.
Day 2 uses OOF probabilities to find the F1-optimal threshold = **0.12**, rescuing the "model has signal but caught nothing" situation.

### Why is test F1 (0.17) lower than OOF F1 (0.26)?

Test set has only **21 Fail samples** — one extra catch or miss shifts F1 by ~5%.
5-fold OOF uses all 83 Fails in train, giving a **more stable estimate**. For interview demos, report the OOF number.

### What to try in Day 3

- Can SMOTE beat `scale_pos_weight`?
- Are LightGBM / CatBoost better suited to high-dim sparse + imbalanced?
- Does keeping Top 100 / all columns dilute signal?

---

## 🏗️ Project Structure

```
semicon-yield-prediction/
├── src/
│   ├── 01_baseline.py        Day 1: XGBoost baseline, connects to PostgreSQL
│   └── 02_improved.py        Day 2: Feature cleaning + OOF threshold tuning
├── notebooks/                Day 3+ exploratory analysis
├── models/                   Trained models (gitignored)
├── docs/                     Screenshots, paper notes
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
Load SECOM data into PostgreSQL first — see the [semicon-yield-dashboard](https://github.com/tzuhua0308/semicon-yield-dashboard) README for full steps.

### Run Baseline and Day 2 Improved
```bash
git clone https://github.com/tzuhua0308/semicon-yield-prediction
cd semicon-yield-prediction
pip install -r requirements.txt
python src/01_baseline.py     # Day 1: F1 = 0.00
python src/02_improved.py     # Day 2: F1 = 0.17 (test) / 0.26 (OOF)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Source | PostgreSQL (shared from sister project, 924K SECOM sensor readings) |
| Data Processing | pandas · numpy · psycopg2 |
| Model | XGBoost · scikit-learn (Day 3 adds LightGBM, CatBoost) |
| Imbalance Handling | scale_pos_weight (Day 2 adds SMOTE / imbalanced-learn) |
| Tuning | Optuna (Day 4) |
| Interpretability | SHAP (Day 4) |
| Demo | Streamlit (Day 5) |

---

## 🎓 Real-World Fab Mapping

| This Project | Real Fab Equivalent |
|---|---|
| Predict Fail | YMS real-time alarm system |
| SHAP finds key sensors | Yield engineer root-cause analysis (RCA) |
| Threshold tuning | "Cast wider net" vs "Only catch high-confidence" business tradeoff |
| SMOTE synthesis | Industry "Rare Event Modeling" |
| Streamlit Demo | Fab IT internal analytics tool |

---

## 📚 Further Reading

- [UCI SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom)
- [Imbalanced-learn documentation](https://imbalanced-learn.org/)
- [SHAP for ML interpretability](https://shap.readthedocs.io/)
