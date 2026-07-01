"""
02_improved.py — SECOM 良率預測改良版

修掉 Day 1 baseline 的 F1=0 問題,四刀齊下:
  1. 去掉 NaN>50% 的感測器欄
  2. Median 補缺
  3. VarianceThreshold 去常數 / 近常數欄
  4. mutual_info_classif 選 Top K 訊號欄
  5. XGBoost + scale_pos_weight
  6. StratifiedKFold 5-fold 在 train 上拿 OOF 機率
  7. 掃 threshold 找 F1 最佳切點(不再用 0.5)

用法:
    python src/02_improved.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DB_URL = os.getenv(
    "SECOM_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/secom",
)

# ─── 超參數(固定,方便重現) ───────────────────────────────────
MISSING_THRESHOLD = 0.50   # NaN 比例超過就砍
TOP_K_FEATURES = 50        # mutual_info 保留幾個特徵
N_SPLITS = 5               # StratifiedKFold folds
RANDOM_STATE = 42

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1,
)


# ─── Step 1: Load ──────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    print("→ Loading SECOM from PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    try:
        df = pd.read_sql(
            """
            SELECT l.lot_id, l.is_pass, s.sensor_code, sd.value
            FROM lots l
            JOIN sensor_data sd ON sd.lot_id = l.lot_id
            JOIN sensors     s  ON s.sensor_id = sd.sensor_id
            """,
            conn,
        )
    finally:
        conn.close()

    X = df.pivot(index="lot_id", columns="sensor_code", values="value")
    y_lookup = df[["lot_id", "is_pass"]].drop_duplicates().set_index("lot_id")
    y = (~y_lookup.loc[X.index, "is_pass"].astype(bool)).astype(int)
    y.name = "label"

    print(f"  X shape: {X.shape}")
    print(f"  y: Pass={int((y == 0).sum())}, Fail={int((y == 1).sum())}")
    return X, y


# ─── Step 2: Preprocess(只 fit 在 train 上避免洩漏) ──────────────
def preprocess(X_train, X_test, y_train):
    print("\n→ Preprocessing...")

    # 2-1 砍 NaN 比例過高的感測器
    miss = X_train.isna().mean()
    keep_missing = miss[miss < MISSING_THRESHOLD].index
    X_train = X_train[keep_missing]
    X_test = X_test[keep_missing]
    print(f"  [1] 砍 NaN>={MISSING_THRESHOLD:.0%} 感測器: {len(miss)} → {len(keep_missing)}")

    # 2-2 median 補缺
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    print("  [2] Median 補缺完成")

    # 2-3 砍常數 / 近常數欄
    vt = VarianceThreshold(threshold=1e-6)
    vt.fit(X_train)
    keep_var = X_train.columns[vt.get_support()]
    X_train = X_train[keep_var]
    X_test = X_test[keep_var]
    print(f"  [3] 砍近常數欄: → {len(keep_var)}")

    # 2-4 mutual_info 選 Top K
    k = min(TOP_K_FEATURES, X_train.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE),
        k=k,
    )
    selector.fit(X_train, y_train)
    keep_mi = X_train.columns[selector.get_support()]
    X_train = X_train[keep_mi]
    X_test = X_test[keep_mi]
    print(f"  [4] mutual_info Top-{k}: → {len(keep_mi)} 個特徵留下")

    return X_train, X_test


# ─── Step 3: OOF + threshold tuning ────────────────────────────────
def find_best_threshold(X_train, y_train, spw) -> tuple[float, float]:
    """用 StratifiedKFold OOF 機率找 F1 最佳 threshold。"""
    print("\n→ StratifiedKFold OOF for threshold tuning...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y_train))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        print(f"  fold {fold}/{N_SPLITS} done")

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = (oof >= t).astype(int)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"  OOF best F1={best_f1:.4f} @ threshold={best_t:.3f}")
    return best_t, best_f1


# ─── Step 4: Train final + evaluate on test ────────────────────────
def evaluate(X_train, X_test, y_train, y_test, spw, threshold):
    print(f"\n→ Final model on full train, evaluate with threshold={threshold:.3f}")
    model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    print("\n" + "═" * 60)
    print("📊 改良版結果(feature select + OOF threshold tuning)")
    print("═" * 60)

    cm = pd.DataFrame(
        confusion_matrix(y_test, pred),
        index=["實際 Pass", "實際 Fail"],
        columns=["預測 Pass", "預測 Fail"],
    )
    print("\nConfusion Matrix:")
    print(cm.to_string())

    print(f"\nF1 (Fail) : {f1_score(y_test, pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_test, proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["Pass", "Fail"], zero_division=0))

    print("── Baseline (Day 1) 對照 ──")
    print("  F1=0.0000  AUC=0.7395  threshold=0.5")


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n→ Train: {len(X_train)} / Test: {len(X_test)}")

    X_train, X_test = preprocess(X_train, X_test, y_train)

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"\n  scale_pos_weight = {spw:.2f}")

    best_t, oof_f1 = find_best_threshold(X_train, y_train, spw)
    evaluate(X_train, X_test, y_train, y_test, spw, best_t)
