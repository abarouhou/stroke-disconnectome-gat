# -*- coding: utf-8 -*-
"""
Extrait et inspecte le fichier cocoyeo143_LUT depuis un des zips NeMo.
Objectif : recuperer les noms anatomiques des 143 regions de l'atlas,
indispensables pour l'interpretabilite (SHAP) cliniquement lisible.

Le LUT sera copie dans nemo_results/cocoyeo143_LUT.txt et son format affiche.
"""

import os
import zipfile

# ============================ A ADAPTER ========================================
ZIPS_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"
OUT_LUT  = "/content/drive/MyDrive/isles_prp/MNI/nemo_results/cocoyeo143_LUT.txt"
# ===============================================================================


def main():
    zip_files = sorted(fn for fn in os.listdir(ZIPS_DIR)
                       if fn.lower().endswith(".zip"))
    if not zip_files:
        print(f"[ERREUR] Aucun .zip dans {ZIPS_DIR}")
        return

    found = None
    for zf in zip_files:
        zpath = os.path.join(ZIPS_DIR, zf)
        try:
            with zipfile.ZipFile(zpath) as z:
                for member in z.namelist():
                    base = os.path.basename(member).lower()
                    # le LUT contient "cocoyeo143" et "lut"
                    if "cocoyeo143" in base and "lut" in base:
                        found = (zf, member)
                        # extraire le contenu
                        with z.open(member) as fh:
                            content = fh.read().decode("utf-8", errors="replace")
                        with open(OUT_LUT, "w", encoding="utf-8") as out:
                            out.write(content)
                        break
        except zipfile.BadZipFile:
            continue
        if found:
            break

    if not found:
        print("[ERREUR] Aucun fichier cocoyeo143_LUT trouve dans les zips.")
        print("Fichiers presents dans le 1er zip (pour reperer le bon nom) :")
        with zipfile.ZipFile(os.path.join(ZIPS_DIR, zip_files[0])) as z:
            for m in z.namelist()[:40]:
                b = os.path.basename(m)
                if b:
                    print("   ", b)
        return

    zf, member = found
    print("=" * 70)
    print(f"LUT trouve dans : {zf}")
    print(f"   membre : {member}")
    print(f"   copie vers : {OUT_LUT}")
    print("=" * 70)

    # Afficher les 20 premieres lignes + compter
    with open(OUT_LUT, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    print(f"\nNombre de lignes : {len(lines)}")
    print("\n--- 20 PREMIERES LIGNES (format brut) ---")
    for i, l in enumerate(lines[:20]):
        # repr() pour voir les tabulations/separateurs
        print(f"{i:>3} | {repr(l)}")

    print("\n--- STRUCTURE DETECTEE ---")
    sample = lines[0] if lines else ""
    if "\t" in sample:
        print("  Separateur probable : TABULATION (\\t)")
        print(f"  Colonnes ligne 0 : {sample.split(chr(9))}")
    elif "," in sample:
        print("  Separateur probable : VIRGULE")
    elif len(sample.split()) > 1:
        print("  Separateur probable : ESPACE(S)")
        print(f"  Champs ligne 0 : {sample.split()}")
    else:
        print("  Un seul champ par ligne (nom de region direct ?)")


if __name__ == "__main__":
    main()
