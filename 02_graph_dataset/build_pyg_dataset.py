# -*- coding: utf-8 -*-
"""
================================================================================
 CHARGEUR PyTorch Geometric — disconnectome ChaCo -> graphes (article 2, GNN)
================================================================================
Transforme les .pkl NeMo (chacoconn 143x143 + chacovol 143) en une liste
d'objets torch_geometric.data.Data, un par patient, prets pour le GAT.

Decisions de design validees :
  1. Cible        : binaire  y = 1 si mRS 3 months > 2 (defavorable), sinon 0
  2. Seuil aretes : proportionnel, on garde le TOP_PERCENT % des aretes (poids)
  3. Features x   : [ChaCo regional (chacovol), degre pondere du noeud]
  4. Graphe       : non-oriente, diagonale = 0 (pas de self-loops)
  5. Normalisation: z-score, fit sur le TRAIN uniquement (fait au moment du
                    split / CV, PAS ici — ici on stocke les features brutes)

Sortie :
  - dataset : list[Data]  (sauvegarde optionnelle en .pt)
  - chaque Data possede : x, edge_index, edge_attr, y, pid
--------------------------------------------------------------------------------
PREREQUIS Colab :
    !pip install torch_geometric
--------------------------------------------------------------------------------
ADAPTER LES CHEMINS puis lancer.
"""

import os
import re
import csv
import io
import zipfile
import pickle
import numpy as np
import torch
from torch_geometric.data import Data

# ============================ A ADAPTER ========================================
MNI_DIR  = "/content/drive/MyDrive/isles_prp/MNI"
ZIPS_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"
OUT_PT   = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"

MRS_COLUMN   = "mRS 3 months"
MRS_THRESHOLD = 2          # y = 1 si mRS > 2  (defavorable)
TOP_PERCENT  = 0.15        # garder le top 15% des aretes par poids ChaCo
# ===============================================================================

PID_RE     = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)
CONN_TOKEN = "chacoconn"
VOL_TOKEN  = "chacovol"
MEAN_TOKEN = "mean.pkl"


# --------------------------------------------------------------------------- #
# 1) Labels mRS (binaires) depuis les outcome.csv                             #
# --------------------------------------------------------------------------- #
def charger_labels(mni_dir):
    labels = {}
    for name in sorted(os.listdir(mni_dir)):
        d = os.path.join(mni_dir, name)
        if not os.path.isdir(d):
            continue
        m = PID_RE.search(name)
        if not m:
            continue
        pid = m.group(1).lower()
        outcome = None
        for fn in os.listdir(d):
            if fn.lower().endswith("outcome.csv"):
                outcome = os.path.join(d, fn)
                break
        if not outcome:
            continue
        with open(outcome, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            row = next(reader, None)
            if not row:
                continue
            raw = (row.get(MRS_COLUMN) or "").strip()
            if raw == "":
                continue
            try:
                mrs = float(raw)
            except ValueError:
                continue
            labels[pid] = 1 if mrs > MRS_THRESHOLD else 0
    return labels


# --------------------------------------------------------------------------- #
# 2) Indexer + charger les .pkl ChaCo depuis les zip (sans extraire)          #
# --------------------------------------------------------------------------- #
def indexer_chaco(zips_dir):
    chaco = {}
    for zf in sorted(fn for fn in os.listdir(zips_dir) if fn.lower().endswith(".zip")):
        zpath = os.path.join(zips_dir, zf)
        try:
            with zipfile.ZipFile(zpath) as z:
                for member in z.namelist():
                    base = os.path.basename(member).lower()
                    if not base.endswith(MEAN_TOKEN):
                        continue
                    m = PID_RE.search(base)
                    if not m:
                        continue
                    pid = m.group(1).lower()
                    slot = chaco.setdefault(pid, {"conn": None, "vol": None})
                    if CONN_TOKEN in base and slot["conn"] is None:
                        slot["conn"] = (zpath, member)
                    elif VOL_TOKEN in base and slot["vol"] is None:
                        slot["vol"] = (zpath, member)
        except zipfile.BadZipFile:
            print(f"[ALERTE] ZIP corrompu ignore : {zf}")
    return chaco


def load_pkl_from_zip(ref):
    """Charge un objet pickle depuis (zip_path, member). Gere sparse/np."""
    zpath, member = ref
    with zipfile.ZipFile(zpath) as z:
        with z.open(member) as fh:
            obj = pickle.load(io.BytesIO(fh.read()))
    # scipy sparse -> dense
    if hasattr(obj, "toarray"):
        obj = obj.toarray()
    return np.asarray(obj, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 3) Matrice ChaCo -> graphe seuille non-oriente                              #
# --------------------------------------------------------------------------- #
def matrice_vers_graphe(conn, top_percent):
    """
    conn : (143,143) matrice de connectivite ChaCo.
    Retourne edge_index (2,E) et edge_attr (E,1) pour un graphe non-oriente,
    diagonale a zero, en gardant le top_percent des aretes (poids les + forts).
    """
    A = np.array(conn, dtype=np.float64)
    n = A.shape[0]

    # symetriser (au cas ou legerement asymetrique) et annuler la diagonale
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    # triangle superieur seulement (graphe non-oriente -> aretes uniques)
    iu, ju = np.triu_indices(n, k=1)
    w = A[iu, ju]

    # ne garder que les poids strictement positifs
    pos = w > 0
    iu, ju, w = iu[pos], ju[pos], w[pos]
    if w.size == 0:
        # graphe vide -> on renvoie un graphe sans aretes (le GAT gerera)
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, 1), dtype=torch.float)
        return ei, ea

    # seuil proportionnel : top_percent des aretes positives
    k = max(1, int(np.ceil(top_percent * w.size)))
    keep = np.argsort(w)[::-1][:k]          # indices des k plus gros poids
    iu, ju, w = iu[keep], ju[keep], w[keep]

    # dedoubler pour non-oriente : (i->j) et (j->i)
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    ew  = np.concatenate([w,  w])

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr  = torch.tensor(ew.reshape(-1, 1),     dtype=torch.float)
    return edge_index, edge_attr


# --------------------------------------------------------------------------- #
# 4) Construire la liste de Data                                              #
# --------------------------------------------------------------------------- #
def build_dataset():
    print("=" * 70)
    print(" CONSTRUCTION DATASET PyG  (ChaCo -> graphes)")
    print("=" * 70)

    labels = charger_labels(MNI_DIR)
    chaco  = indexer_chaco(ZIPS_DIR)
    print(f"Labels mRS charges        : {len(labels)}")
    print(f"Patients avec ChaCo       : {len(chaco)}")

    dataset, skipped = [], []
    for pid in sorted(labels):
        c = chaco.get(pid)
        if not c or not c["conn"] or not c["vol"]:
            skipped.append(pid)
            continue

        conn = load_pkl_from_zip(c["conn"])           # (143,143)
        vol  = load_pkl_from_zip(c["vol"]).reshape(-1) # (143,) apres squeeze

        n = conn.shape[0]
        if vol.shape[0] != n:
            print(f"[ALERTE] {pid}: dim vol {vol.shape[0]} != n {n}, ignore")
            skipped.append(pid)
            continue

        edge_index, edge_attr = matrice_vers_graphe(conn, TOP_PERCENT)

        # features de noeuds : [ChaCo regional, degre pondere]
        # degre pondere = somme des poids des aretes incidentes (sur graphe seuille)
        deg = np.zeros(n, dtype=np.float64)
        if edge_index.numel() > 0:
            src = edge_index[0].numpy()
            w   = edge_attr.squeeze(-1).numpy()
            np.add.at(deg, src, w)

        x = np.vstack([vol, deg]).T                   # (143, 2)
        x = torch.tensor(x, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([labels[pid]], dtype=torch.long),
        )
        data.pid = pid
        data.num_nodes = n
        dataset.append(data)

    print(f"\nGraphes construits        : {len(dataset)}")
    if skipped:
        print(f"Patients ignores (no ChaCo): {len(skipped)} -> {skipped}")

    # Stats rapides
    if dataset:
        ys = np.array([int(d.y) for d in dataset])
        n1 = int(ys.sum()); n0 = len(ys) - n1
        e  = np.array([d.edge_index.shape[1] // 2 for d in dataset])  # aretes uniques
        print("\n--- STATISTIQUES DATASET ---")
        print(f"  Classe 0 (mRS<=2, favorable)   : {n0}")
        print(f"  Classe 1 (mRS>2, defavorable)  : {n1}")
        print(f"  Ratio classes                  : {n0}/{n1} "
              f"({100*n1/len(ys):.1f}% defavorable)")
        print(f"  Aretes/graphe (apres seuil)    : "
              f"min={e.min()} median={int(np.median(e))} max={e.max()}")
        print(f"  Noeuds/graphe                  : {dataset[0].num_nodes}")
        print(f"  Dim features noeud (x)         : {dataset[0].x.shape[1]}")

    return dataset


# --------------------------------------------------------------------------- #
# MAIN                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ds = build_dataset()
    if ds:
        torch.save(ds, OUT_PT)
        print(f"\nDataset sauvegarde : {OUT_PT}")
        print(f"   Rechargement ulterieur : ds = torch.load('{OUT_PT}')")
        print("\n--- EXEMPLE (1er graphe) ---")
        d0 = ds[0]
        print(f"  pid        : {d0.pid}")
        print(f"  x          : {tuple(d0.x.shape)}")
        print(f"  edge_index : {tuple(d0.edge_index.shape)}")
        print(f"  edge_attr  : {tuple(d0.edge_attr.shape)}")
        print(f"  y          : {int(d0.y)}")
