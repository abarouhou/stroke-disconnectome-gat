# -*- coding: utf-8 -*-
"""
================================================================================
 GAT — BLOC 2 : ENTRAINEMENT + CV 5x10  (article 2, GNN disconnectome)
================================================================================
Entraine le StrokeGAT (bloc 1) selon le MEME protocole que les baselines, pour
une comparaison equitable :
   - CV stratifiee 5-fold repetee 10x (memes seeds/splits que baselines)
   - test de chaque fold INTOUCHABLE (jamais vu a l'entrainement)
   - early stopping PROPRE : split interne train/val DANS le train du fold
   - class_weight (BCEWithLogits pos_weight) pour le ratio 77/50
   - normalisation clinique z-score + imputation : fit sur TRAIN-INTERNE only
   - collecte des predictions OUT-OF-FOLD (pour tests stats du bloc 3)

Enrichit d'abord chaque graphe avec data.clinical (18 vars, memes que B4).

Sortie : dict de resultats par config + predictions poolees -> gat_results.pt
--------------------------------------------------------------------------------
PREREQUIS : bloc 1 (StrokeGAT) execute dans une cellule precedente.
"""

import os
import re
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

# ============================ A ADAPTER ========================================
MNI_DIR = "/content/drive/MyDrive/isles_prp/MNI"
PT_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
OUT_PT  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/gat_results.pt"

N_SPLITS, N_REPEATS, SEED = 5, 10, 42
MAX_EPOCHS = 200
PATIENCE   = 20          # early stopping
LR         = 5e-3
WEIGHT_DECAY = 5e-4      # regularisation L2 (petit dataset)
BATCH_SIZE = 16
VAL_FRAC   = 0.2         # part du train du fold reservee a la validation interne
# ===============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)

# 18 variables cliniques (identiques baseline B4, vides 100% retirees)
CLIN_NUM = ["Age", "Glucose", "Leucocytes", "CRP", "INR",
            "NIHSS at admission", "mRS premorbid"]
CLIN_BIN = ["Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia",
            "Anticoagulation", "Lipid lowering drugs", "PAIs",
            "Wake-up", "In-House", "Referral"]
CLIN_COLS = CLIN_NUM + CLIN_BIN + ["Sex_M"]   # 18 colonnes


# --------------------------------------------------------------------------- #
# 1) Enrichir les graphes avec les variables cliniques (brutes, non normees)  #
# --------------------------------------------------------------------------- #
def enrich_clinical(dataset, mni_dir):
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

    n_ok = 0
    for d in dataset:
        pid = d.pid.lower()
        r = rows.get(pid, {})
        vals = []
        for c in CLIN_NUM + CLIN_BIN:
            vals.append(tf(r.get(c, "")))
        sx = (r.get("Sex", "") or "").upper()
        vals.append(1.0 if sx == "M" else (0.0 if sx == "F" else np.nan))
        d.clinical_raw = torch.tensor(vals, dtype=torch.float).unsqueeze(0)  # (1,18)
        if not np.isnan(vals).all():
            n_ok += 1
    print(f"Graphes enrichis clinique : {n_ok}/{len(dataset)}")
    return dataset


# --------------------------------------------------------------------------- #
# 2) Normalisation clinique fit sur TRAIN-INTERNE (imputation + z-score)      #
# --------------------------------------------------------------------------- #
def fit_clinical_stats(dataset, idx):
    """Calcule medianes (imputation) et mean/std (z-score) sur les indices idx."""
    M = np.vstack([dataset[i].clinical_raw.numpy() for i in idx])  # (n,18)
    med = np.nanmedian(M, axis=0)
    med = np.where(np.isnan(med), 0.0, med)             # colonne 100% NaN -> 0
    Mi = np.where(np.isnan(M), med, M)
    mean = Mi.mean(axis=0)
    std = Mi.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return med, mean, std


def apply_clinical(dataset, med, mean, std):
    """Ecrit data.clinical (imputee + normee) sur tout le dataset."""
    for d in dataset:
        v = d.clinical_raw.numpy().copy()
        v = np.where(np.isnan(v), med, v)
        v = (v - mean) / std
        d.clinical = torch.tensor(v, dtype=torch.float)   # (1,18)


# --------------------------------------------------------------------------- #
# 3) Entrainement d'un fold avec early stopping sur validation interne        #
# --------------------------------------------------------------------------- #
def train_one_fold(dataset, tr_idx, te_idx, use_clinical, seed):
    # split interne train/val (stratifie) DANS le train du fold
    y_tr = np.array([int(dataset[i].y) for i in tr_idx])
    tin_idx, val_idx = train_test_split(
        tr_idx, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)

    # stats cliniques fit sur TRAIN-INTERNE uniquement
    if use_clinical:
        med, mean, std = fit_clinical_stats(dataset, tin_idx)
        apply_clinical(dataset, med, mean, std)

    train_set = [dataset[i] for i in tin_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in te_idx]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=64)
    test_loader  = DataLoader(test_set,  batch_size=64)

    # pos_weight pour le desequilibre (sur le train-interne)
    n1 = int(y_tr.sum()); n0 = len(y_tr) - n1
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    torch.manual_seed(seed)
    model = StrokeGAT(use_clinical=use_clinical).to(DEVICE)   # StrokeGAT du bloc 1
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        # --- train
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            opt.step()
        # --- val (AUC)
        val_auc = eval_auc(model, val_loader)
        if val_auc > best_val:
            best_val, best_state, wait = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # restaurer le meilleur etat, predire le TEST une seule fois
    if best_state is not None:
        model.load_state_dict(best_state)
    probs, ys = predict(model, test_loader)
    return probs, ys, te_idx


@torch.no_grad()
def eval_auc(model, loader):
    model.eval()
    ps, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        p = torch.sigmoid(model(batch)).cpu().numpy()
        ps.append(p); ys.append(batch.y.cpu().numpy())
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    if len(np.unique(ys)) < 2:
        return 0.5
    return roc_auc_score(ys, ps)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        p = torch.sigmoid(model(batch)).cpu().numpy()
        ps.append(p); ys.append(batch.y.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# 4) CV 5x10 pour une config donnee                                          #
# --------------------------------------------------------------------------- #
def run_cv(dataset, use_clinical, tag):
    y = np.array([int(d.y) for d in dataset])
    X_dummy = np.zeros((len(y), 1))
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    fold_aucs = []
    # predictions out-of-fold moyennees par repetition
    oof_accum = np.zeros(len(y)); oof_count = np.zeros(len(y))

    print(f"\n>>> Config: {tag} (use_clinical={use_clinical})")
    for k, (tr, te) in enumerate(rskf.split(X_dummy, y)):
        seed_k = SEED + k
        probs, ys, te_idx = train_one_fold(dataset, tr, te, use_clinical, seed_k)
        auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else 0.5
        fold_aucs.append(auc)
        oof_accum[te_idx] += probs
        oof_count[te_idx] += 1
        if (k + 1) % N_SPLITS == 0:
            rep = (k + 1) // N_SPLITS
            m = np.mean(fold_aucs[-N_SPLITS:])
            print(f"   repetition {rep:2d}/{N_REPEATS}  AUC={m:.3f}")

    fold_aucs = np.array(fold_aucs)
    oof_probs = oof_accum / np.maximum(oof_count, 1)
    print(f"   => {tag}: AUC = {fold_aucs.mean():.3f} ± {fold_aucs.std():.3f}")
    return {"tag": tag, "use_clinical": use_clinical,
            "fold_aucs": fold_aucs, "oof_probs": oof_probs, "y": y}


# --------------------------------------------------------------------------- #
# MAIN                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print(f" GAT — CV {N_SPLITS}x{N_REPEATS} | device={DEVICE}")
    print("=" * 70)

    dataset = torch.load(PT_PATH, weights_only=False)
    print(f"Graphes charges : {len(dataset)}")
    dataset = enrich_clinical(dataset, MNI_DIR)

    results = {}
    # Deux bras cle pour l'ablation (bloc 3 ajoutera les comparaisons stats)
    results["GAT_graphe_seul"] = run_cv(dataset, use_clinical=False,
                                        tag="GAT graphe-seul")
    results["GAT_clinique"]    = run_cv(dataset, use_clinical=True,
                                        tag="GAT + clinique")

    torch.save(results, OUT_PT)
    print(f"\nResultats sauvegardes : {OUT_PT}")
    print("\n--- RESUME ---")
    for k, r in results.items():
        a = r["fold_aucs"]
        print(f"  {r['tag']:<22} AUC = {a.mean():.3f} ± {a.std():.3f}")
    print("\n  (rappel baseline B4 ChaCo+Clin XGB = 0.836)")


if __name__ == "__main__":
    main()
