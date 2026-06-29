"""
01_baseline.py — SECOM 良率預測 baseline

從 PostgreSQL 撈 SECOM 寬表 → XGBoost 二分類 → 印出 metrics。

Pass=0(良品)、Fail=1(不良,我們關注的 positive class)。

用法:
    python src/01_baseline.py
"""

import os
import warnings

import pandas as pd
import psycopg2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DB_URL = os.getenv(
    "SECOM_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/secom",
)


# ─── Step 1: Load ──────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """從 Postgres 撈寬表 X (1567, 590) 與 y (1567,)。"""
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

    # long → wide
    X = df.pivot(index="lot_id", columns="sensor_code", values="value")

    # y:Fail=1(關注的 positive class)、Pass=0
    y_lookup = df[["lot_id", "is_pass"]].drop_duplicates().set_index("lot_id")
    y = (~y_lookup.loc[X.index, "is_pass"].astype(bool)).astype(int)
    y.name = "label"

    print(f"  X shape: {X.shape}")
    print(f"  y: Pass={int((y == 0).sum())}, Fail={int((y == 1).sum())}")
    print(f"  不平衡比: {(y == 0).sum() / max((y == 1).sum(), 1):.1f}:1")
    return X, y


# ─── Step 2: Train ─────────────────────────────────────────────────
def train_xgb(X: pd.DataFrame, y: pd.Series):
    """80/20 stratified split + baseline XGBoost。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 用 scale_pos_weight 處理 14:1 不平衡
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"\n→ Train: {len(X_train)} / Test: {len(X_test)}")
    print(f"  scale_pos_weight = {spw:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ─── Step 3: Evaluate ──────────────────────────────────────────────
def evaluate(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "═" * 56)
    print("📊 Baseline 結果(XGBoost,SECOM 寬表 590 特徵,無 NaN 處理)")
    print("═" * 56)

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["實際 Pass", "實際 Fail"],
        columns=["預測 Pass", "預測 Fail"],
    )
    print("\nConfusion Matrix:")
    print(cm.to_string())

    print(f"\nF1 (Fail) : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_data()
    model, X_test, y_test = train_xgb(X, y)
    evaluate(model, X_test, y_test)
