# -*- coding: utf-8 -*-
"""
================================================================================
 GAT — BLOC 3b : EXTRACTION DE L'ATTENTION  (article 2, GNN disconnectome)
 [VERSION CORRIGEE — bug tensor GPU/CPU resolu]
================================================================================
Produit la CONTRIBUTION INTERPRETABLE de l'article : quelles regions cerebrales
(atlas CocoYeo143) le GAT regarde pour predire un pronostic defavorable.

Methode (robuste, pas un seul modele) :
   1. Entraine N_MODELS GAT+clinique sur des splits differents (stabilite).
   2. Pour chaque modele, extrait l'attention des 2 couches (return_attention).
   3. Agrege les poids d'ARETES en score d'importance par NOEUD (region) :
      importance(region) = somme de l'attention des aretes incidentes.
   4. Moyenne sur les N_MODELS + separe par classe (defavorable y=1 / favorable).
   5. Sort une table triee des regions les plus attentionnees + un .npy.
--------------------------------------------------------------------------------
PREREQUIS : bloc 1 (StrokeGAT) execute. torch_geometric OK. pyg_dataset.pt present.
"""

import os
import re
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

# ============================ A ADAPTER ========================================
MNI_DIR = "/content/drive/MyDrive/isles_prp/MNI"
PT_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
OUT_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"
# Optionnel : chemin d'un cocoyeo143_LUT (1 nom/ligne ou "index<TAB>nom").
# Laisser "" -> regions nommees R000..R142
LUT_PATH = ""

N_MODELS = 10
MAX_EPOCHS = 200
PATIENCE = 20
LR = 5e-3
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 16
VAL_FRAC = 0.2
SEED = 42
# ===============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)

CLIN_NUM = ["Age", "Glucose", "Leucocytes", "CRP", "INR",
            "NIHSS at admission", "mRS premorbid"]
CLIN_BIN = ["Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia",
            "Anticoagulation", "Lipid lowering drugs", "PAIs",
            "Wake-up", "In-House", "Referral"]


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

    for d in dataset:
        rr = rows.get(d.pid.lower(), {})
        vals = [tf(rr.get(c, "")) for c in CLIN_NUM + CLIN_BIN]
        sx = (rr.get("Sex", "") or "").upper()
        vals.append(1.0 if sx == "M" else (0.0 if sx == "F" else np.nan))
        d.clinical_raw = torch.tensor(vals, dtype=torch.float).unsqueeze(0)
    return dataset


def fit_apply_clinical(dataset, idx_fit):
    # .cpu() defensif : clinical_raw doit rester lisible par numpy
    M = np.vstack([dataset[i].clinical_raw.cpu().numpy() for i in idx_fit])
    med = np.nanmedian(M, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Mi = np.where(np.isnan(M), med, M)
    mean, std = Mi.mean(0), Mi.std(0); std = np.where(std < 1e-8, 1.0, std)
    for d in dataset:
        v = d.clinical_raw.cpu().numpy().copy()
        v = np.where(np.isnan(v), med, v)
        d.clinical = torch.tensor((v - mean) / std, dtype=torch.float)


def load_lut(path, n=143):
    if path and os.path.exists(path):
        names = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                names.append(parts[-1])
        if len(names) >= n:
            return names[:n]
    return [f"R{i:03d}" for i in range(n)]


def train_one_model(dataset, tr_idx, seed, n_nodes=143):
    y_tr = np.array([int(dataset[i].y) for i in tr_idx])
    tin, val = train_test_split(tr_idx, test_size=VAL_FRAC,
                                stratify=y_tr, random_state=seed)
    fit_apply_clinical(dataset, tin)

    train_loader = DataLoader([dataset[i] for i in tin],
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val], batch_size=64)

    n1 = int(y_tr.sum()); n0 = len(y_tr) - n1
    pos_w = torch.tensor([n0 / max(n1, 1)], dtype=torch.float, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    torch.manual_seed(seed)
    model = StrokeGAT(use_clinical=True).to(DEVICE)   # bloc 1
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best, best_state, wait = -1, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(batch), batch.y.float())
            loss.backward(); opt.step()
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                ps.append(torch.sigmoid(model(batch)).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
        ps, ys = np.concatenate(ps), np.concatenate(ys)
        va = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
        if va > best:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)

    # --- Extraction attention par classe, agregee au niveau noeud ---
    # IMPORTANT : on ne fait PLUS dataset[i].to(DEVICE) (contaminait le CPU).
    # Batch.from_data_list cree une COPIE qu'on envoie sur GPU pour le forward.
    model.eval()
    imp = {0: np.zeros(n_nodes), 1: np.zeros(n_nodes)}
    cnt = {0: 0, 1: 0}
    with torch.no_grad():
        for i in range(len(dataset)):
            b = Batch.from_data_list([dataset[i]]).to(DEVICE)
            _, att1, att2 = model(b, return_attention=True)
            node_score = np.zeros(n_nodes)
            for att in (att1, att2):
                ei, alpha = att
                ei = ei.cpu().numpy()
                a = alpha.mean(dim=1).cpu().numpy()   # moyenne sur les tetes
                np.add.at(node_score, ei[1], a)       # attention recue (dest)
            yi = int(dataset[i].y)
            imp[yi] += node_score
            cnt[yi] += 1
    for c in (0, 1):
        if cnt[c] > 0:
            imp[c] /= cnt[c]
    return imp[0], imp[1], best


def main():
    print("=" * 72)
    print(f" EXTRACTION ATTENTION — {N_MODELS} modeles | device={DEVICE}")
    print("=" * 72)

    dataset = torch.load(PT_PATH, weights_only=False)
    dataset = enrich_clinical(dataset, MNI_DIR)
    n_nodes = dataset[0].x.shape[0]
    y = np.array([int(d.y) for d in dataset])
    names = load_lut(LUT_PATH, n_nodes)

    idx = np.arange(len(dataset))
    skf = StratifiedKFold(n_splits=N_MODELS, shuffle=True, random_state=SEED)

    fav = np.zeros((N_MODELS, n_nodes))
    unfav = np.zeros((N_MODELS, n_nodes))
    aucs = []
    for k, (tr, _) in enumerate(skf.split(idx, y)):
        s0, s1, va = train_one_model(dataset, tr, SEED + k, n_nodes)
        fav[k], unfav[k] = s0, s1
        aucs.append(va)
        print(f"  modele {k+1:2d}/{N_MODELS}  val_AUC={va:.3f}")

    fav_mean, unfav_mean = fav.mean(0), unfav.mean(0)
    fav_std, unfav_std = fav.std(0), unfav.std(0)
    diff = unfav_mean - fav_mean

    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else x
    unfav_n = norm(unfav_mean)

    order = np.argsort(unfav_mean)[::-1]
    out_csv = os.path.join(OUT_DIR, "attention_regions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rang", "region_idx", "region_nom",
                    "attention_defavorable", "attention_defav_norm",
                    "std_defav", "attention_favorable",
                    "diff_defav_moins_fav"])
        for rang, i in enumerate(order, 1):
            w.writerow([rang, i, names[i],
                        f"{unfav_mean[i]:.5f}", f"{unfav_n[i]:.4f}",
                        f"{unfav_std[i]:.5f}", f"{fav_mean[i]:.5f}",
                        f"{diff[i]:+.5f}"])

    np.save(os.path.join(OUT_DIR, "attention_scores.npy"),
            {"favorable": fav, "unfavorable": unfav, "names": names})

    print("\n" + "-" * 72)
    print(f"val_AUC moyen des {N_MODELS} modeles : "
          f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print("\nTOP 15 regions les plus attentionnees (pronostic defavorable) :")
    print(f"{'rang':>4} {'idx':>4} {'region':<20} {'attn':>8} {'diff':>8}")
    for rang, i in enumerate(order[:15], 1):
        print(f"{rang:>4} {i:>4} {names[i]:<20} "
              f"{unfav_mean[i]:>8.4f} {diff[i]:>+8.4f}")

    print("\n" + "-" * 72)
    print(f"CSV ecrit  : {out_csv}")
    print(f"NPY ecrit  : {os.path.join(OUT_DIR, 'attention_scores.npy')}")
    print("\nA soumettre aux co-auteurs cliniciens pour validation :")
    print("  -> les regions du TOP sont-elles coherentes avec le pronostic AVC ?")
    print("  -> 'diff' positif = region specifiquement regardee pour le mauvais")
    print("     pronostic (vs favorable).")


if __name__ == "__main__":
    main()
