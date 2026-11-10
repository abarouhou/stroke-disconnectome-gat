# Graph attention networks on structural disconnectomes for stroke outcome prediction

Code accompanying the article:

> Barouhou A, Machkour N, Tibar H, Regragui W, Benhlima L, Bah S. *Does graph topology matter? End-to-end graph attention networks on structural disconnectomes for functional outcome prediction after ischemic stroke.* Submitted to *Neuroinformatics*.

The pipeline predicts unfavorable 90-day functional outcome (mRS > 2) in 127 patients of the public **ISLES 2024** dataset from (i) structural disconnectomes computed with **NeMo** on the CocoYeo143 atlas and (ii) 18 admission-only clinical variables. It compares an end-to-end graph attention network (GAT) with logistic regression and XGBoost under repeated stratified 5×10 cross-validation, tests the contribution of graph topology (implicit and explicit ablations, DeLong tests) and explains predictions with SHAP.

**No patient data are included in this repository.** ISLES 2024 must be obtained from the challenge organizers (https://isles-24.grand-challenge.org/), and disconnectomes must be computed with NeMo (https://github.com/kjamison/nemo).

## Repository structure

| Folder | Content | Article section |
|---|---|---|
| `01_preprocessing/` | Native → MNI lesion registration with ANTsPy (SyN) and QC; inventory/QC of NeMo outputs; extraction and alignment check of the CocoYeo143 lookup table | 3.1–3.2 |
| `02_graph_dataset/` | NeMo ChaCo outputs → PyTorch Geometric graphs (143 nodes, top-15% edges, node features [regional ChaCo, weighted degree]) | 3.4 |
| `03_baselines/` | Logistic regression and XGBoost baselines B1–B4 | 3.6, 4.2 |
| `04_gat/` | GAT model, training (5×10 CV, internal validation, early stopping), DeLong tests (C1–C3), full metrics and ROC curves | 3.5, 3.7, 4.3–4.4 |
| `05_topology_ablation/` | Explicit topological node features (betweenness, eigenvector centrality, clustering, PageRank) and topology-enriched GAT | 3.8, 4.5 |
| `06_interpretability/` | Stabilized SHAP analysis (regions, functional networks); GAT attention analysis (Appendix A) | 3.9, 4.6, App. A |
| `07_figures_tables/` | Figures (atlas, lesion overlap, patient graph, group maps, SHAP brain maps) and Table 1 / registration QC | 4.1, 4.6 |

## Order of execution

1. `01_preprocessing/registration_native_to_MNI.ipynb` → lesion masks in MNI space, then upload them to the NeMo web service (iFOD2+ACT, SIFT2 weighting, CocoYeo143 subject-specific parcellation, pairwise output).
2. `01_preprocessing/nemo_inventory_qc.py`, `extract_lut.py`, `verify_lut_alignment.py`
3. `02_graph_dataset/build_pyg_dataset.py` → `pyg_dataset.pt`
4. `03_baselines/baselines.py`
5. `04_gat/gat_model.py` then `04_gat/gat_training.py` (run in the same session/notebook) → `gat_results.pt`; then `delong_tests.py` and `gat_metrics_roc.ipynb`
6. `05_topology_ablation/topology_ablation.ipynb`
7. `06_interpretability/shap_analysis.py`
8. `07_figures_tables/*.ipynb`

The scripts were written for Google Colab with data stored on Google Drive. Edit the path constants at the top of each script (`BASE`, `MNI_DIR`, `NEMO`, `PT_PATH`, …) to match your own directory layout. Code comments are partly in French.

## Main settings

- Cross-validation: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`, identical splits for all models.
- GAT: 2 × `GATConv` (4 heads × 64, then 1 head × 64), mean‖max readout, clinical branch (18 → 16), MLP head; Adam (lr 5e-3, weight decay 5e-4), batch 16, dropout 0.4, max 200 epochs, early stopping on internal validation AUC (patience 20, 20% of the training fold).
- XGBoost: 300 trees, max depth 3, learning rate 0.05, subsample 0.8, colsample 0.8.
- Preprocessing (median imputation, z-scoring) fitted on training folds only.

## Software

Python 3.13, PyTorch 2.11, PyTorch Geometric 2.8.1, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.54.1, NetworkX 3.7, nilearn, ANTsPy (see `requirements.txt`).

## Citation

If you use this code, please cite the article above (citation to be updated upon publication).

## License

MIT (see `LICENSE`).
