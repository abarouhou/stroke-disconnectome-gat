# -*- coding: utf-8 -*-
"""
================================================================================
 BLOC 3c CORRIGE : INTERPRETABILITE SHAP (stable + sens correct)
================================================================================
Corrige la v1 sur deux points :
  1. SENS directionnel : calcule par CORRELATION entre la valeur de la feature
     et sa valeur SHAP (spearman). Le "signe moyen" de la v1 etait trompeur pour
     un modele non-lineaire. Ici : correlation > 0 => la feature ELEVEE pousse
     vers le pronostic defavorable (mRS>2).
  2. STABILITE : SHAP moyenne sur N_SPLITS modeles (calcul sur le test OOF de
     chaque fold). Donne importance +/- ecart-type => on distingue signal stable
     et bruit de tirage.

Interprete XGBoost B4 (ChaCo+clinique), le modele le plus performant.
Deux niveaux : regions (143) + reseaux de Yeo.
--------------------------------------------------------------------------------
PREREQUIS : pip install shap. region_names.txt present.
"""

import os
import re
import csv
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

# ============================ A ADAPTER ========================================
MNI_DIR = "/content/drive/MyDrive/isles_prp/MNI"
PT_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
NAMES_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/region_names.txt"
OUT_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"
N_SPLITS = 5
N_REPEATS = 4          # 5x4 = 20 modeles pour stabiliser
SEED = 42
# ===============================================================================

CLIN_NUM = ["Age", "Glucose", "Leucocytes", "CRP", "INR",
            "NIHSS at admission", "mRS premorbid"]
CLIN_BIN = ["Atrial fibrillation", "Hypertension", "Diabetes", "Hyperlipidemia",
            "Anticoagulation", "Lipid lowering drugs", "PAIs",
            "Wake-up", "In-House", "Referral"]
CLIN_COLS = CLIN_NUM + CLIN_BIN + ["Sex_M"]
PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)


def load_region_names(path, n=143):
    with open(path, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    assert len(names) == n, f"{len(names)} != {n}"
    return names


def yeo_network(name):
    if name.startswith("Cbm_"):
        return "Cerebellum (SUIT)"
    subcort = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus",
               "Amygdala", "Accumbens", "VentralDC")
    if any(s in name for s in subcort):
        return "Subcortical (ASEG)"
    m = re.search(r"7Networks_([A-Za-z]+)", name)
    if m:
        mapping = {"Vis": "Visual", "SomMot": "Somatomotor",
                   "DorsAttn": "DorsalAttention",
                   "SalVentAttn": "SalienceVentAttn", "Limbic": "Limbic",
                   "Cont": "Control", "Default": "Default"}
        return mapping.get(m.group(1), m.group(1))
    return "Other"


def load_data():
    ds = torch.load(PT_PATH, weights_only=False)
    pids = [d.pid.lower() for d in ds]
    y = np.array([int(d.y) for d in ds])
    X_chaco = np.vstack([d.x[:, 0].numpy() for d in ds])
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
    return pids, y, X_chaco, np.array(clin)


def main():
    try:
        import shap
    except ImportError:
        print("[ERREUR] pip install shap"); return
    from xgboost import XGBClassifier

    print("=" * 74)
    print(f" SHAP STABILISE ({N_SPLITS}x{N_REPEATS} modeles) — sens corrige")
    print("=" * 74)

    region_names = load_region_names(NAMES_PATH)
    pids, y, X_chaco, X_clin = load_data()
    X = np.hstack([X_chaco, X_clin])
    n_chaco = len(region_names)
    n_feat = X.shape[1]

    imp_glob = SimpleImputer(strategy="median")
    X_full_imp = imp_glob.fit_transform(X)   # pour le calcul de correlation sens

    # accumulateurs SHAP OOF
    shap_oof = np.zeros((len(y), n_feat))
    shap_cnt = np.zeros(len(y))
    imp_per_model = []  # |shap| moyen par feature, par modele (pour std)

    n1 = int(y.sum()); n0 = len(y) - n1
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr]); Xte = imp.transform(X[te])
        model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=SEED,
                              scale_pos_weight=n0 / max(n1, 1))
        model.fit(Xtr, y[tr])
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xte)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        shap_oof[te] += sv
        shap_cnt[te] += 1
        imp_per_model.append(np.abs(sv).mean(axis=0))
        if (k + 1) % N_SPLITS == 0:
            print(f"  repetition {(k+1)//N_SPLITS}/{N_REPEATS} ok")

    shap_oof /= np.maximum(shap_cnt, 1)[:, None]
    imp_per_model = np.array(imp_per_model)       # (n_models, n_feat)

    mean_abs = np.abs(shap_oof).mean(axis=0)      # importance
    std_abs = imp_per_model.std(axis=0)           # stabilite entre modeles

    # --- SENS par correlation de Spearman feature <-> shap ---
    sens_corr = np.zeros(n_feat)
    for j in range(n_feat):
        xj = X_full_imp[:, j]
        if np.std(xj) < 1e-9:
            sens_corr[j] = 0.0
        else:
            rho, _ = spearmanr(xj, shap_oof[:, j])
            sens_corr[j] = 0.0 if np.isnan(rho) else rho

    def sens_txt(rho):
        if rho > 0.1:
            return "AGGRAVE (val haute -> defavorable)"
        if rho < -0.1:
            return "PROTEGE (val haute -> favorable)"
        return "ambigu/non-lineaire"

    chaco_imp = mean_abs[:n_chaco]; chaco_std = std_abs[:n_chaco]
    chaco_sens = sens_corr[:n_chaco]
    clin_imp = mean_abs[n_chaco:]; clin_sens = sens_corr[n_chaco:]

    tot_c, tot_cl = chaco_imp.sum(), clin_imp.sum()
    print("\n--- CONTRIBUTION GLOBALE (SHAP OOF stabilise) ---")
    print(f"  Disconnectome : {100*tot_c/(tot_c+tot_cl):.1f}%")
    print(f"  Clinique      : {100*tot_cl/(tot_c+tot_cl):.1f}%")

    # Niveau regions
    order = np.argsort(chaco_imp)[::-1]
    with open(os.path.join(OUT_DIR, "shap_regions_v2.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rang", "region", "reseau", "shap_abs", "shap_std",
                    "spearman_sens", "interpretation"])
        for rang, i in enumerate(order, 1):
            w.writerow([rang, region_names[i], yeo_network(region_names[i]),
                        f"{chaco_imp[i]:.5f}", f"{chaco_std[i]:.5f}",
                        f"{chaco_sens[i]:+.3f}", sens_txt(chaco_sens[i])])

    print("\n--- TOP 15 REGIONS (importance stable + sens corrige) ---")
    print(f"{'rg':>3} {'region':<32} {'|shap|':>7} {'std':>6} {'rho':>6} sens")
    for rang, i in enumerate(order[:15], 1):
        print(f"{rang:>3} {region_names[i]:<32} {chaco_imp[i]:>7.4f} "
              f"{chaco_std[i]:>6.4f} {chaco_sens[i]:>+6.2f} "
              f"{sens_txt(chaco_sens[i])}")

    # Niveau reseaux
    nets, nets_rho, nets_n = {}, {}, {}
    for i, rn in enumerate(region_names):
        net = yeo_network(rn)
        nets[net] = nets.get(net, 0.0) + chaco_imp[i]
        nets_rho[net] = nets_rho.get(net, 0.0) + chaco_sens[i]
        nets_n[net] = nets_n.get(net, 0) + 1
    net_order = sorted(nets, key=nets.get, reverse=True)
    with open(os.path.join(OUT_DIR, "shap_reseaux_v2.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rang", "reseau", "shap_abs_total",
                    "spearman_moyen", "interpretation"])
        for rang, net in enumerate(net_order, 1):
            rho_moy = nets_rho[net] / nets_n[net]
            w.writerow([rang, net, f"{nets[net]:.5f}",
                        f"{rho_moy:+.3f}", sens_txt(rho_moy)])

    print("\n--- RESEAUX DE YEO (importance stable + sens moyen) ---")
    print(f"{'rg':>3} {'reseau':<20} {'|shap|':>8} {'rho_moy':>8} sens")
    for rang, net in enumerate(net_order, 1):
        rho_moy = nets_rho[net] / nets_n[net]
        print(f"{rang:>3} {net:<20} {nets[net]:>8.4f} {rho_moy:>+8.2f} "
              f"{sens_txt(rho_moy)}")

    # Clinique
    corder = np.argsort(clin_imp)[::-1]
    print("\n--- TOP variables CLINIQUES (sens corrige) ---")
    for rang, i in enumerate(corder[:8], 1):
        print(f"{rang:>3} {CLIN_COLS[i]:<20} {clin_imp[i]:>7.4f} "
              f"{clin_sens[i]:>+6.2f} {sens_txt(clin_sens[i])}")

    np.save(os.path.join(OUT_DIR, "shap_values_v2.npy"),
            {"shap_oof": shap_oof, "sens_corr": sens_corr,
             "region_names": region_names, "y": y})

    print("\n" + "-" * 74)
    print("CSV : shap_regions_v2.csv / shap_reseaux_v2.csv")
    print("\nControle de plausibilite clinique :")
    print("  - Age doit etre AGGRAVE (rho>0). NIHSS aussi si present.")
    print("  - 'std' faible = region stable ; 'std' ~ |shap| = instable (prudence).")


if __name__ == "__main__":
    main()
