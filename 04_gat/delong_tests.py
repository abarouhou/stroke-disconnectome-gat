# -*- coding: utf-8 -*-
"""
================================================================================
 GAT — BLOC 3a : TESTS STATISTIQUES (DeLong)  (article 2, GNN disconnectome)
================================================================================
Compare les AUC out-of-fold (OOF) des modeles DEUX A DEUX sur les MEMES patients
et les MEMES splits, via le test de DeLong (comparaison de deux courbes ROC
correlees). Repond a la question du jury : "la difference est-elle significative
ou dans le bruit ?".

Comparaisons cle :
   C1. GAT graphe-seul   vs  ChaCo-vecteur (XGB)   -> la topologie apporte-t-elle qqch ?
   C2. GAT + clinique     vs  ChaCo+Clin (XGB=B4)   -> le GAT complet est-il inferieur ?
   C3. GAT + clinique     vs  Clinique seule        -> le disconnectome ajoute-t-il ?

Principe :
   - on recharge les predictions OOF du GAT (gat_results.pt, bloc 2)
   - on regenere les predictions OOF des baselines sur les MEMES splits (SEED=42)
   - DeLong donne p-value + IC de la difference d'AUC
--------------------------------------------------------------------------------
PREREQUIS : numpy, scipy, sklearn, xgboost (deja installes). gat_results.pt present.
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ============================ A ADAPTER ========================================
MNI_DIR = "/content/drive/MyDrive/isles_prp/MNI"
PT_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
GAT_PT  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/gat_results.pt"
OUT_CSV = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/delong_tests.csv"

N_SPLITS, N_REPEATS, SEED = 5, 10, 42
MRS_THRESHOLD = 2
# ===============================================================================

PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)
CLIN_NUM = ["Age", "Glucose", "Leucocytes", "CRP", "INR",
            "NIHSS at admission", "mRS premorbid"]
CLIN_BIN = ["Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia",
            "Anticoagulation", "Lipid lowering drugs", "PAIs",
            "Wake-up", "In-House", "Referral"]


# --------------------------------------------------------------------------- #
# Implementation DeLong (comparaison de 2 AUC correlees, meme echantillon)     #
#   Retourne : auc1, auc2, difference, z, p-value (bilaterale), IC95 de diff   #
#   Base sur DeLong et al. 1988 (algo rapide de Sun & Xu 2014).                #
# --------------------------------------------------------------------------- #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(preds_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, score1, score2):
    """Compare deux scores sur le meme y_true. Retourne dict."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")  # positifs d'abord
    label_1_count = int(y_true.sum())
    preds = np.vstack((score1, score2))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    auc1, auc2 = float(aucs[0]), float(aucs[1])
    diff = auc1 - auc2
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(var) if var > 0 else 1e-12
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return {"auc1": auc1, "auc2": auc2, "diff": diff,
            "z": float(z), "p": float(p), "ci": ci}


# --------------------------------------------------------------------------- #
# Chargement des donnees (alignees sur l'ordre du dataset PyG)                 #
# --------------------------------------------------------------------------- #
def load_data():
    ds = torch.load(PT_PATH, weights_only=False)
    pids = [d.pid.lower() for d in ds]
    y = np.array([int(d.y) for d in ds])
    X_chaco = np.vstack([d.x[:, 0].numpy() for d in ds])  # (n,143)

    rows = {}
    for name in os.listdir(MNI_DIR):
        d = os.path.join(MNI_DIR, name)
        if not os.path.isdir(d):
            continue
        m = PID_RE.search(name)
        if not m:
            continue
        pid = m.group(1).lower()
        f = next((os.path.join(d, fn) for fn in os.listdir(d)
                  if fn.lower().endswith("demographic_baseline.csv")), None)
        if not f:
            continue
        with open(f, newline="", encoding="utf-8-sig") as fh:
            r = csv.DictReader(fh)
            r.fieldnames = [h.strip() for h in r.fieldnames]
            row = next(r, None)
            if row:
                rows[pid] = {k.strip(): (v.strip() if v else "")
                             for k, v in row.items()}

    def tf(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    clin = []
    for pid in pids:
        rr = rows.get(pid, {})
        feat = [tf(rr.get(c, "")) for c in CLIN_NUM + CLIN_BIN]
        sx = (rr.get("Sex", "") or "").upper()
        feat.append(1.0 if sx == "M" else (0.0 if sx == "F" else np.nan))
        clin.append(feat)
    X_clin = np.array(clin)
    return pids, y, X_chaco, X_clin


def make_logreg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000,
                                                class_weight="balanced"))])


def make_xgb():
    clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=SEED)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])


def oof_predictions(X, y, model_fn):
    """Predictions out-of-fold moyennees sur les repetitions, memes splits."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    accum = np.zeros(len(y)); count = np.zeros(len(y))
    for tr, te in rskf.split(X, y):
        model = model_fn()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        accum[te] += p; count[te] += 1
    return accum / np.maximum(count, 1)


# --------------------------------------------------------------------------- #
# MAIN                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 74)
    print(" TESTS DE DELONG — comparaison des AUC out-of-fold (memes splits)")
    print("=" * 74)

    pids, y, X_chaco, X_clin = load_data()
    X_chaco_clin = np.hstack([X_chaco, X_clin])

    # Predictions OOF du GAT (bloc 2)
    gat = torch.load(GAT_PT, weights_only=False)
    oof_gat_graph = gat["GAT_graphe_seul"]["oof_probs"]
    oof_gat_clin  = gat["GAT_clinique"]["oof_probs"]
    y_gat = gat["GAT_clinique"]["y"]
    assert np.array_equal(y, y_gat), "Desalignement labels GAT vs dataset !"

    # Predictions OOF des baselines (memes splits)
    print("\nRegeneration des predictions OOF des baselines...")
    oof_chaco_xgb = oof_predictions(X_chaco, y, make_xgb)       # ChaCo vecteur
    oof_clin_lr   = oof_predictions(X_clin,  y, make_logreg)    # Clinique seule
    oof_b4_xgb    = oof_predictions(X_chaco_clin, y, make_xgb)  # B4 ChaCo+Clin

    def auc(p):
        return roc_auc_score(y, p)

    print("\n--- AUC out-of-fold (poolees) ---")
    print(f"  GAT graphe-seul   : {auc(oof_gat_graph):.3f}")
    print(f"  GAT + clinique    : {auc(oof_gat_clin):.3f}")
    print(f"  ChaCo-vecteur XGB : {auc(oof_chaco_xgb):.3f}")
    print(f"  Clinique LR       : {auc(oof_clin_lr):.3f}")
    print(f"  B4 ChaCo+Clin XGB : {auc(oof_b4_xgb):.3f}")

    comparisons = [
        ("C1 GAT-graphe vs ChaCo-vecteur", oof_gat_graph, oof_chaco_xgb),
        ("C2 GAT+clin vs B4 (ChaCo+Clin)", oof_gat_clin,  oof_b4_xgb),
        ("C3 GAT+clin vs Clinique seule",  oof_gat_clin,  oof_clin_lr),
    ]

    print("\n" + "-" * 74)
    print(" RESULTATS DELONG")
    print("-" * 74)
    rows_out = []
    for name, s1, s2 in comparisons:
        r = delong_test(y, s1, s2)
        signif = "SIGNIFICATIF" if r["p"] < 0.05 else "non significatif"
        print(f"\n{name}")
        print(f"   AUC1={r['auc1']:.3f}  AUC2={r['auc2']:.3f}  "
              f"diff={r['diff']:+.3f}")
        print(f"   IC95 diff=[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
              f"z={r['z']:+.2f}  p={r['p']:.3f}  -> {signif}")
        rows_out.append([name, f"{r['auc1']:.4f}", f"{r['auc2']:.4f}",
                         f"{r['diff']:+.4f}", f"{r['ci'][0]:+.4f}",
                         f"{r['ci'][1]:+.4f}", f"{r['z']:+.3f}",
                         f"{r['p']:.4f}", signif])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparaison", "AUC1", "AUC2", "diff_AUC",
                    "IC95_bas", "IC95_haut", "z", "p_value", "conclusion"])
        w.writerows(rows_out)

    print("\n" + "-" * 74)
    print(f"CSV ecrit : {OUT_CSV}")
    print("\nLECTURE :")
    print("  C1 non significatif -> la topologie du graphe n'ajoute rien vs vecteur.")
    print("  C2 non significatif -> GAT+clin EQUIVALENT a XGBoost (non inferieur).")
    print("  C3 significatif      -> le disconnectome ajoute a la clinique seule.")


if __name__ == "__main__":
    main()
