# -*- coding: utf-8 -*-
"""
================================================================================
 PISTE 1 — Features topologiques explicites (article 2, contribution CS)
================================================================================
Objectif : forcer le GAT a exploiter la STRUCTURE du disconnectome, pas seulement
la charge de deconnexion regionale. On enrichit les features de noeuds avec des
descripteurs TOPOLOGIQUES issus de la forme du graphe (pas de l'intensite ChaCo) :

   - ChaCo regional        (conserve : l'atteinte de la region)
   - degre pondere         (conserve)
   - betweenness centrality (le noeud est-il un carrefour du reseau ?)
   - eigenvector centrality (est-il connecte a des noeuds influents ?)
   - clustering coefficient (ses voisins sont-ils relies entre eux ?)
   - pagerank               (importance globale dans le graphe)

Ces 4 mesures topologiques ne sont PAS deductibles du vecteur ChaCo plat :
si elles ameliorent la prediction, c'est la topologie qui apporte l'information.

Sortie : pyg_dataset_topo.pt  (memes 127 patients, x de dimension 6 au lieu de 2)
=> a passer aux blocs 2 (GAT) et 3a (DeLong) EXISTANTS, en changeant PT_PATH.
--------------------------------------------------------------------------------
PREREQUIS : torch, torch_geometric, networkx (preinstalle sur Colab).
"""

import os
import numpy as np
import torch
import networkx as nx

# ============================ A ADAPTER ========================================
PT_IN  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset.pt"
PT_OUT = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/pyg_dataset_topo.pt"
# ===============================================================================


def node_topo_features(edge_index, edge_attr, n_nodes):
    """
    Calcule les descripteurs topologiques par noeud a partir du graphe seuille.
    Retourne un array (n_nodes, 4) : [betweenness, eigenvector, clustering, pagerank].
    Les poids d'aretes servent de force de connexion.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if edge_index.numel() > 0:
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        w = edge_attr.squeeze(-1).numpy() if edge_attr is not None else np.ones(len(src))
        for s, d, wt in zip(src, dst, w):
            if s < d:  # graphe non-oriente : une arete par paire
                G.add_edge(int(s), int(d), weight=float(wt))

    # betweenness (carrefours) — poids = distance inverse du poids fort
    try:
        # distance = 1/poids : une connexion forte = un chemin "court"
        for u, v, dd in G.edges(data=True):
            dd["dist"] = 1.0 / (dd["weight"] + 1e-6)
        btw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    except Exception:
        btw = {i: 0.0 for i in range(n_nodes)}

    # eigenvector centrality (connecte a des noeuds influents)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {i: 0.0 for i in range(n_nodes)}

    # clustering pondere (voisins relies entre eux)
    try:
        clu = nx.clustering(G, weight="weight")
    except Exception:
        clu = {i: 0.0 for i in range(n_nodes)}

    # pagerank (importance globale)
    try:
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {i: 1.0/n_nodes for i in range(n_nodes)}
    except Exception:
        pr = {i: 1.0/n_nodes for i in range(n_nodes)}

    feats = np.zeros((n_nodes, 4), dtype=np.float64)
    for i in range(n_nodes):
        feats[i, 0] = btw.get(i, 0.0)
        feats[i, 1] = eig.get(i, 0.0)
        feats[i, 2] = clu.get(i, 0.0)
        feats[i, 3] = pr.get(i, 0.0)
    return feats


def main():
    print("=" * 70)
    print(" PISTE 1 — Enrichissement topologique des features de noeuds")
    print("=" * 70)

    dataset = torch.load(PT_IN, weights_only=False)
    print(f"Graphes charges : {len(dataset)}")
    n_nodes = dataset[0].x.shape[0]

    empty_graphs = 0
    for k, d in enumerate(dataset):
        # x existant : [ChaCo regional, degre] (n_nodes, 2)
        x_old = d.x.numpy()
        topo = node_topo_features(d.edge_index, d.edge_attr, n_nodes)
        if d.edge_index.numel() == 0:
            empty_graphs += 1
        # nouvelle matrice de features : (n_nodes, 6)
        x_new = np.hstack([x_old, topo])
        d.x = torch.tensor(x_new, dtype=torch.float)
        if (k + 1) % 30 == 0:
            print(f"  {k+1}/{len(dataset)} patients traites")

    # normalisation par colonne des 4 nouvelles features (z-score global)
    # NB : pour l'entrainement, la vraie normalisation train-only se fait dans
    # le bloc 2 ; ici on laisse les valeurs brutes pour rester coherent.
    print(f"\nGraphes vides (0 arete) : {empty_graphs}")
    print(f"Nouvelle dim features de noeud : {dataset[0].x.shape[1]} "
          f"(etait 2, maintenant 6)")
    print("  [0]=ChaCo regional  [1]=degre  [2]=betweenness  "
          "[3]=eigenvector  [4]=clustering  [5]=pagerank")

    # apercu sur 1 patient
    d0 = dataset[0]
    print(f"\nExemple {d0.pid} — stats des features topologiques :")
    for j, name in enumerate(["betweenness", "eigenvector", "clustering", "pagerank"]):
        col = d0.x[:, 2 + j].numpy()
        print(f"   {name:<14} min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f}")

    torch.save(dataset, PT_OUT)
    print(f"\nDataset topologique sauvegarde : {PT_OUT}")
    print("\n=== SUITE ===")
    print("  1. Dans le BLOC 1 (StrokeGAT), passer in_channels=6 au lieu de 2 :")
    print("     model = StrokeGAT(in_channels=6, use_clinical=...)")
    print("  2. Dans le BLOC 2, changer PT_PATH -> pyg_dataset_topo.pt et relancer.")
    print("  3. Dans le BLOC 3a (DeLong), comparer le NOUVEAU GAT-graphe au")
    print("     ChaCo-vecteur : si p<0.05 en faveur du GAT, la topologie AIDE.")


if __name__ == "__main__":
    main()
