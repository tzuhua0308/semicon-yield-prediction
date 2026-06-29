# Semicon Yield Prediction

> 半導體製程良率預測模型 — 用 SECOM 感測器資料預測晶圓批號 Pass / Fail,
> 並用 SHAP 找出關鍵影響因子。

> 🔗 姊妹專案:[semicon-yield-dashboard](https://github.com/tzuhua0308/semicon-yield-dashboard)
> (本 repo 直接連到該專案的 PostgreSQL,共享同一份 SECOM 資料)

---

## 🎯 專案目標

從 590 個製程感測器讀值預測「**這批晶圓會不會 Fail**」,業界對應到:
- 在離開機台 **5 分鐘內** 預判可能 Fail 的 lot,提早送量測站重檢
- 找出**對良率影響最大**的感測器(SHAP),協助工程師排查根因

---

## 📦 資料來源

| 項目 | 內容 |
|---|---|
| 資料集 | UCI SECOM Dataset(透過姊妹專案的 PostgreSQL 共享) |
| 樣本數 | 1,567 批晶圓 |
| 特徵數 | 590 個感測器讀值 |
| 標籤 | Pass(良品)=0,Fail(不良)=1 |
| **類別比** | **14.1 : 1**(極度不平衡)⚠️ |
| 缺失值 | 4.54% NaN |

---

## 🗺️ Roadmap

- [x] **Day 1**(6/29):Baseline XGBoost — F1 = 0.00, AUC = 0.74(展示「Majority Class Predictor」陷阱)
- [ ] **Day 2**:特徵工程(去除 NULL>80% 感測器、中位數補缺、mutual_info 特徵選擇)+ SMOTE 平衡
- [ ] **Day 3**:四模型對戰(RF / XGBoost / LightGBM / CatBoost)
- [ ] **Day 4**:Optuna 貝氏調參 + SHAP 可解釋性(找出 Top 10 影響良率的感測器)
- [ ] **Day 5**:Streamlit Demo(輸入感測器值 → 即時 Pass/Fail 預測)
- [ ] **Day 6-7**:README 完善、截圖、推 GitHub

---

## 📊 Day 1 Baseline 結果

```
Confusion Matrix:
              預測 Pass   預測 Fail
實際 Pass       291         2
實際 Fail        21         0    ← 21 個 Fail 全部沒抓到!

F1 (Fail) : 0.0000   ❌ 完全沒抓到不良品
ROC AUC   : 0.7395   ✅ 模型有訊號,只是 threshold 不對
Accuracy  : 93%      ⚠️ 假象(因為猜全部 Pass 就有 93%)
```

### 學到的事(刻意展示)

**這個 baseline 看起來「失敗」,但揭露了 3 個重要觀念**:

1. **不平衡分類絕對不能看 Accuracy** — 「全部猜多數類」就有 93%,但實際上沒做事。
2. **AUC 0.74 表示模型有訊號** — 只是預設 threshold = 0.5 太保守。
3. **這是業界的 "Yield Excursion early warning" 痛點** — Fail 樣本天生稀少,
   是不平衡學習(Imbalanced Learning)的經典場景。

Day 2 會用 **SMOTE + Feature Selection + Threshold Tuning** 把 F1 拉到 ~0.30+。

---

## 🏗️ 專案結構

```
semicon-yield-prediction/
├── src/
│   └── 01_baseline.py        Day 1: XGBoost baseline,連 PostgreSQL
├── notebooks/                Day 2+ 探索式分析
├── models/                   訓練好的模型(gitignore)
├── docs/                     截圖、論文筆記
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 快速開始

### 前置作業
先把 [semicon-yield-dashboard](https://github.com/tzuhua0308/semicon-yield-dashboard)
裡的 SECOM 資料載入 PostgreSQL(那邊的 README 有完整步驟)。

### 跑 Baseline
```bash
git clone https://github.com/tzuhua0308/semicon-yield-prediction
cd semicon-yield-prediction
pip install -r requirements.txt
python src/01_baseline.py
```

---

## 🛠️ 技術棧

| 層 | 工具 |
|---|---|
| 資料來源 | PostgreSQL(從姊妹專案共享 SECOM 924K 筆讀值) |
| 資料處理 | pandas · numpy · psycopg2 |
| 模型 | XGBoost · scikit-learn(Day 3 加 LightGBM、CatBoost) |
| 不平衡處理 | scale_pos_weight(Day 2 加 SMOTE / imbalanced-learn) |
| 調參 | Optuna(Day 4) |
| 可解釋性 | SHAP(Day 4) |
| Demo | Streamlit(Day 5) |

---

## 🎓 業界場景對照

| 本專案 | 真實 fab 廠對應 |
|---|---|
| 預測 Fail | YMS 即時 alarm 系統 |
| SHAP 找關鍵感測器 | 良率工程師根因分析(RCA) |
| Threshold tuning | 「寧錯殺不放過」vs「不抓到不要」的業務取捨 |
| SMOTE 合成 | 對應業界「Rare Event Modeling」 |
| Streamlit Demo | 對應 fab IT 部門內部分析工具 |

---

## 📚 相關閱讀

- [UCI SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom)
- [Imbalanced-learn 文件](https://imbalanced-learn.org/)
- [SHAP for ML interpretability](https://shap.readthedocs.io/)
