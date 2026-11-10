# -*- coding: utf-8 -*-
"""
================================================================================
 BASELINES v2 — version archivee proprement (article 2, GNN disconnectome)
================================================================================
Identique aux baselines validees, avec deux ameliorations :
  - Retrait des 2 variables cliniques VIDES a 100% (Onset to door, Alert to door)
  - Sauvegarde reproductible des resultats :
        * baselines_results.csv    (metriques, pour Excel / article)
        * baselines_context.json   (config + versions + cohorte + date)
        * baselines_table.tex      (tableau LaTeX pret a coller)

Protocole inchange : CV stratifiee 5-fold x10, AUC-ROC principale,
z-score + imputation fit train-only, class_weight balanced.
"""

import os
import re
import csv
import json
import platform
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score)
import sklearn

try:
    from xgboost import XGBClassifier
    import xgboost
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ============================ A ADAPTER ========================================
MNI_DIR  = "/content/drive/MyDrive/isles_prp/MNI"
PT_PATH  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
OUT_DIR  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/baselines_archive"

MRS_COLUMN, MRS_THRESHOLD = "mRS 3 months", 2
N_SPLITS, N_REPEATS, SEED = 5, 10, 42
# ===============================================================================

os.makedirs(OUT_DIR, exist_ok=True)
PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)

# Variables ADMISSION (sans fuite). VIDES 100% retirees : Onset/Alert to door.
CLIN_NUM = ["Age", "Glucose", "Leucocytes", "CRP", "INR",
            "NIHSS at admission", "mRS premorbid"]
CLIN_BIN = ["Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia",
            "Anticoagulation", "Lipid lowering drugs", "PAIs",
            "Wake-up", "In-House", "Referral"]


def load_pyg_order(pt_path):
    import torch
    ds = torch.load(pt_path, weights_only=False)
    pids, y, vol = [], [], []
    for d in ds:
        pids.append(d.pid.lower())
        y.append(int(d.y))
        vol.append(float(d.x[:, 0].sum()))
    return pids, np.array(y), np.array(vol).reshape(-1, 1), ds


def load_chaco_matrix(ds):
    return np.vstack([d.x[:, 0].numpy() for d in ds])


def load_clinical(mni_dir, pids):
    rows = {}
    for name in os.listdir(mni_dir):
        d = os.path.join(mni_dir, name)
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

    data = []
    for pid in pids:
        r = rows.get(pid, {})
        feat = {c: tf(r.get(c, "")) for c in CLIN_NUM + CLIN_BIN}
        sx = (r.get("Sex", "") or "").upper()
        feat["Sex_M"] = 1.0 if sx == "M" else (0.0 if sx == "F" else np.nan)
        data.append(feat)
    return pd.DataFrame(data, index=pids)


def make_logreg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000,
                                                class_weight="balanced"))])


def make_gb():
    if HAS_XGB:
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=SEED)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state=SEED)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])


def evaluate(X, y, model_fn, name):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    aucs, aps, baccs, sens, spec = [], [], [], [], []
    for tr, te in rskf.split(X, y):
        model = model_fn()
        model.fit(X[tr], y[tr])
        p = (model.predict_proba(X[te])[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X[te]))
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], pred))
        sens.append(recall_score(y[te], pred, pos_label=1, zero_division=0))
        spec.append(recall_score(y[te], pred, pos_label=0, zero_division=0))
    return {"name": name,
            "AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
            "AP_mean": float(np.mean(aps)),
            "bACC_mean": float(np.mean(baccs)),
            "Sens_mean": float(np.mean(sens)),
            "Spec_mean": float(np.mean(spec))}


def main():
    print("=" * 78)
    print(" BASELINES v2 — archivage reproductible")
    print("=" * 78)

    pids, y, vol, ds = load_pyg_order(PT_PATH)
    X_chaco = load_chaco_matrix(ds)
    clin_df = load_clinical(MNI_DIR, pids)
    X_clin = clin_df.values
    X_chaco_clin = np.hstack([X_chaco, X_clin])

    n1 = int(y.sum()); n0 = len(y) - n1
    print(f"Patients {len(y)} | classe0={n0} classe1={n1} "
          f"({100*n1/len(y):.1f}% defavorable)")
    print(f"Clinique {X_clin.shape[1]} vars (2 vides retirees) | "
          f"ChaCo {X_chaco.shape[1]}")

    results = [
        evaluate(X_clin,        y, make_logreg, "B1 Clinique [LogReg]"),
        evaluate(vol,           y, make_logreg, "B2 Volume lesion [LogReg]"),
        evaluate(X_chaco,       y, make_logreg, "B3 ChaCo 143 [LogReg]"),
        evaluate(X_chaco,       y, make_gb,     "B3 ChaCo 143 [XGB]"),
        evaluate(X_chaco_clin,  y, make_logreg, "B4 ChaCo+Clin [LogReg]"),
        evaluate(X_chaco_clin,  y, make_gb,     "B4 ChaCo+Clin [XGB]"),
    ]
    results.sort(key=lambda d: d["AUC_mean"])

    print("\n--- RESULTATS ---")
    for r in results:
        print(f"  {r['name']:<28} AUC={r['AUC_mean']:.3f}±{r['AUC_std']:.3f}  "
              f"bACC={r['bACC_mean']:.3f}  Sens={r['Sens_mean']:.3f}  "
              f"Spec={r['Spec_mean']:.3f}")
    best = max(results, key=lambda d: d["AUC_mean"])
    print(f"\n>>> CIBLE GAT : {best['name']} AUC={best['AUC_mean']:.3f}")

    # --- 1) CSV ---
    csv_path = os.path.join(OUT_DIR, "baselines_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modele", "AUC_mean", "AUC_std", "AP_mean",
                    "bACC_mean", "Sens_mean", "Spec_mean"])
        for r in results:
            w.writerow([r["name"], f"{r['AUC_mean']:.4f}", f"{r['AUC_std']:.4f}",
                        f"{r['AP_mean']:.4f}", f"{r['bACC_mean']:.4f}",
                        f"{r['Sens_mean']:.4f}", f"{r['Spec_mean']:.4f}"])

    # --- 2) JSON contexte ---
    ctx = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "cohorte": {"n": len(y), "classe0_favorable": n0,
                    "classe1_defavorable": n1,
                    "pct_defavorable": round(100*n1/len(y), 1)},
        "cible": {"colonne": MRS_COLUMN, "seuil": MRS_THRESHOLD,
                  "definition": "y=1 si mRS>2 (defavorable)"},
        "protocole": {"cv": f"RepeatedStratifiedKFold {N_SPLITS}x{N_REPEATS}",
                      "seed": SEED, "metrique_principale": "AUC-ROC",
                      "normalisation": "z-score fit train-only",
                      "imputation": "mediane fit train-only",
                      "class_weight": "balanced"},
        "features": {"clinique_vars": CLIN_NUM + CLIN_BIN + ["Sex_M"],
                     "vides_retirees": ["Onset to door", "Alert to door"],
                     "chaco_dim": int(X_chaco.shape[1]),
                     "fuite_exclue": ["mRS at admission", "TICI",
                                      "door-to-* delays", "NIHSS/mRS 24h",
                                      "NIHSS/mRS discharge"]},
        "versions": {"python": platform.python_version(),
                     "sklearn": sklearn.__version__,
                     "xgboost": (xgboost.__version__ if HAS_XGB else "absent"),
                     "numpy": np.__version__, "pandas": pd.__version__},
        "resultats": results,
        "meilleure_baseline": {"nom": best["name"],
                               "AUC": round(best["AUC_mean"], 4),
                               "cible_a_battre_par_GAT": True},
    }
    json_path = os.path.join(OUT_DIR, "baselines_context.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)

    # --- 3) LaTeX ---
    tex_path = os.path.join(OUT_DIR, "baselines_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tableau baselines - article 2 (GNN disconnectome)\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Performance des baselines pour la prediction du "
                "pronostic fonctionnel (mRS $>$ 2) a 3 mois. "
                f"Cohorte n={len(y)}, validation croisee stratifiee "
                f"{N_SPLITS}-fold repetee {N_REPEATS} fois.}}\n")
        f.write("\\label{tab:baselines}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Modele & AUC & AP & bACC & Sens. & Spec. \\\\\n\\hline\n")
        for r in results:
            bold = r["name"] == best["name"]
            nm = r["name"].replace("&", "\\&")
            line = (f"{nm} & {r['AUC_mean']:.3f}$\\pm${r['AUC_std']:.3f} & "
                    f"{r['AP_mean']:.3f} & {r['bACC_mean']:.3f} & "
                    f"{r['Sens_mean']:.3f} & {r['Spec_mean']:.3f}")
            f.write(("\\textbf{" + line + "}" if bold else line) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print("\n--- FICHIERS ECRITS ---")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()
