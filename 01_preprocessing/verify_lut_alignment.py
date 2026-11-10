# -*- coding: utf-8 -*-
"""
Parse cocoyeo143_LUT (format FreeSurfer : index nom R G B A) et construit le
mapping index_python(0..142) -> nom anatomique, en sautant l'index 0 (Unknown).

Verifie l'alignement avec la matrice ChaCo :
  - 143 regions attendues
  - affiche le mapping par blocs (aseg / SUIT / Schaefer) pour controle visuel
  - sauve region_names.txt (1 nom par ligne, ordre = colonnes ChaCo)
--------------------------------------------------------------------------------
IMPORTANT : ce script suppose que l'ordre des colonnes ChaCo == ordre du LUT
(index 1..143). On le VERIFIE ici avant de l'utiliser pour SHAP.
"""

import os
import re
import zipfile
import io
import pickle
import numpy as np

# ============================ A ADAPTER ========================================
LUT_PATH = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/cocoyeo143_LUT.txt"
ZIPS_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"
OUT_NAMES = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/region_names.txt"
# ===============================================================================


def parse_lut(path):
    """Retourne dict {index_lut:int -> nom:str} pour les vraies regions (index>=1)."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            mapping[idx] = name
    return mapping


def main():
    lut = parse_lut(LUT_PATH)
    print("=" * 70)
    print(f"LUT parse : {len(lut)} entrees (index 0 inclus)")

    # Regions reelles = index 1..143 (on saute 0 = Unknown)
    max_idx = max(lut.keys())
    region_idx = [i for i in range(1, max_idx + 1) if i in lut]
    names = [lut[i] for i in region_idx]
    print(f"Regions reelles (index>=1) : {len(names)}")
    print(f"Index 0 = '{lut.get(0, '?')}' (exclu, = fond)")

    if len(names) != 143:
        print(f"\n[ALERTE] {len(names)} noms != 143 attendus. A investiguer.")
    else:
        print("\n[OK] 143 regions, conforme a l'atlas CocoYeo143.")

    # Apercu par blocs
    print("\n--- BLOCS ANATOMIQUES (controle de l'ordre) ---")
    print("Python[0..15]  (ASEG sous-cortical attendu) :")
    for p in range(0, 16):
        print(f"   chaco[{p:>3}] = LUT#{region_idx[p]:>3} = {names[p]}")
    print("Python[16..20] (SUIT cervelet attendu) :")
    for p in range(16, 21):
        print(f"   chaco[{p:>3}] = LUT#{region_idx[p]:>3} = {names[p]}")
    print("Python[43..47] (Schaefer cortical attendu apres cervelet) :")
    for p in range(43, 48):
        print(f"   chaco[{p:>3}] = LUT#{region_idx[p]:>3} = {names[p]}")
    print("Python[138..142] (dernieres regions) :")
    for p in range(138, 143):
        print(f"   chaco[{p:>3}] = LUT#{region_idx[p]:>3} = {names[p]}")

    # Verif croisee avec un chacovol temoin
    print("\n--- VERIF CROISEE avec un chacovol reel ---")
    pid_re = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)
    vol = None; pid_used = None
    for zf in sorted(fn for fn in os.listdir(ZIPS_DIR) if fn.endswith(".zip")):
        with zipfile.ZipFile(os.path.join(ZIPS_DIR, zf)) as z:
            for m in z.namelist():
                b = os.path.basename(m).lower()
                if "chacovol" in b and b.endswith("mean.pkl"):
                    with z.open(m) as fh:
                        obj = pickle.load(io.BytesIO(fh.read()))
                    if hasattr(obj, "toarray"):
                        obj = obj.toarray()
                    vol = np.asarray(obj, dtype=float).reshape(-1)
                    pid_used = pid_re.search(b).group(1)
                    break
        if vol is not None:
            break

    if vol is None:
        print("  [ALERTE] aucun chacovol trouve pour verifier.")
    else:
        print(f"  Patient temoin : {pid_used}  | dim chacovol = {vol.shape[0]}")
        if vol.shape[0] != len(names):
            print(f"  [ALERTE] dim {vol.shape[0]} != {len(names)} noms !")
        else:
            print(f"  [OK] dimensions alignees ({vol.shape[0]} = {len(names)})")
            top = np.argsort(vol)[::-1][:10]
            print("  Top 10 regions les plus deconnectees chez ce patient :")
            for r in top:
                print(f"     chaco[{r:>3}] = {names[r]:<32} vol={vol[r]:.4f}")
            print("  -> Ces regions sont-elles coherentes avec une lesion typique ?")
            print("     (si oui, l'alignement LUT<->ChaCo est valide)")

    # Sauvegarde des noms dans l'ordre ChaCo
    with open(OUT_NAMES, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    print(f"\nNoms sauvegardes (ordre ChaCo) : {OUT_NAMES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
