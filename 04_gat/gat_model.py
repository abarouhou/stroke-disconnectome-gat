# -*- coding: utf-8 -*-
"""
================================================================================
 GAT — BLOC 1 : ARCHITECTURE  (article 2, GNN disconnectome)
================================================================================
Modele GAT end-to-end pour predire P(mRS > 2) a partir du graphe disconnectome
ChaCo, avec fusion optionnelle des variables cliniques.

Concu pour les DEUX angles de la contribution :
  - PERFORMANCE   : architecture compacte, regularisee (n=127 petit)
  - INTERPRETABILITE : chaque couche GAT peut renvoyer ses coefficients
                       d'attention (return_attention_weights) -> mapping
                       clinique ulterieur sur les regions CocoYeo143.

Concu pour la GRILLE D'ABLATION (bloc 3) via le flag use_clinical :
  - use_clinical=False -> GAT GRAPHE-SEUL   (teste l'apport de la topologie)
  - use_clinical=True  -> GAT + CLINIQUE    (modele complet)

Ossature (validee) :
  INPUT graphe (143 noeuds, x=[ChaCo, degre])
    -> GAT couche 1 (multi-tete, ELU, dropout)
    -> GAT couche 2
    -> Readout global (mean + max pooling)  = vecteur graphe
    -> [FUSION avec clinique projetee]      (si use_clinical)
    -> MLP -> logit -> sigmoide -> P(mRS>2)
--------------------------------------------------------------------------------
PREREQUIS : torch, torch_geometric (deja installes dans le notebook).
Ce bloc DEFINIT le modele ; l'entrainement est dans le bloc 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class StrokeGAT(nn.Module):
    """
    GAT 2 couches + readout mean/max + fusion clinique optionnelle + MLP.

    Parametres
    ----------
    in_channels    : dim features de noeud (ici 2 : [ChaCo regional, degre])
    hidden         : dim cachee par tete (defaut 64)
    heads          : nb de tetes d'attention couche 1 (defaut 4)
    clinical_dim   : nb de variables cliniques (ici 18) ; ignore si use_clinical=False
    clinical_proj  : dim de projection de la clinique avant fusion (defaut 16)
    dropout        : taux de dropout (defaut 0.4 ; eleve car petit dataset)
    use_clinical   : True -> fusionne la clinique ; False -> graphe seul (ablation)
    """

    def __init__(self, in_channels=2, hidden=64, heads=4,
                 clinical_dim=18, clinical_proj=16,
                 dropout=0.4, use_clinical=True):
        super().__init__()
        self.use_clinical = use_clinical
        self.dropout = dropout

        # --- Couche GAT 1 : multi-tete, sorties concatenees -> hidden*heads
        self.gat1 = GATConv(in_channels, hidden, heads=heads,
                            dropout=dropout, add_self_loops=True)
        # --- Couche GAT 2 : ramene a 'hidden', 1 tete (moyennee), pour le readout
        self.gat2 = GATConv(hidden * heads, hidden, heads=1,
                            concat=False, dropout=dropout, add_self_loops=True)

        # Readout mean+max -> vecteur graphe de dim 2*hidden
        graph_dim = hidden * 2

        # --- Projection clinique (seulement si fusion)
        if use_clinical:
            self.clin_proj = nn.Sequential(
                nn.Linear(clinical_dim, clinical_proj),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            head_in = graph_dim + clinical_proj
        else:
            self.clin_proj = None
            head_in = graph_dim

        # --- Tete de prediction (MLP -> 1 logit)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data, return_attention=False):
        """
        data : Batch PyG (x, edge_index, batch) + data.clinical (B, clinical_dim)
        return_attention : si True, renvoie aussi les (edge_index, alpha) des 2 couches
        Retour : logits (B,) [, att1, att2]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        att1 = att2 = None
        if return_attention:
            x, att1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
        else:
            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.gat2(x, edge_index))

        # Readout : concat mean + max pooling sur les noeuds de chaque graphe
        g = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)

        # Fusion clinique optionnelle
        if self.use_clinical:
            c = self.clin_proj(data.clinical)
            g = torch.cat([g, c], dim=1)

        logits = self.head(g).squeeze(-1)

        if return_attention:
            return logits, att1, att2
        return logits


# --------------------------------------------------------------------------- #
# Test rapide de coherence des dimensions (facultatif, a lancer une fois)     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    # 2 faux graphes facticies pour verifier que tout passe
    def fake_graph(n=143, in_ch=2, clin=18):
        x = torch.randn(n, in_ch)
        ei = torch.randint(0, n, (2, 500))
        d = Data(x=x, edge_index=ei, y=torch.tensor([0]))
        d.clinical = torch.randn(1, clin)
        return d

    batch = Batch.from_data_list([fake_graph(), fake_graph()])

    print("=== Test GAT + clinique ===")
    m = StrokeGAT(use_clinical=True)
    out = m(batch)
    print("logits:", tuple(out.shape), "(attendu (2,))")

    print("\n=== Test GAT graphe-seul (ablation) ===")
    m0 = StrokeGAT(use_clinical=False)
    out0 = m0(batch)
    print("logits:", tuple(out0.shape), "(attendu (2,))")

    print("\n=== Test sortie attention ===")
    logits, a1, a2 = m(batch, return_attention=True)
    print("att couche 1 : edge_index", tuple(a1[0].shape),
          "| alpha", tuple(a1[1].shape))
    print("att couche 2 : edge_index", tuple(a2[0].shape),
          "| alpha", tuple(a2[1].shape))
    print("\nOK — architecture coherente.")
