# -*- coding: utf-8 -*-
"""
================================================================================
 INVENTAIRE QC — NeMo / ChaCo  ↔  cohorte mRS (article 2, GNN disconnectome)
================================================================================
Objectif :
  1. Parcourir tous les dossiers patients MNI/sub-stroke* et construire la liste
     des INCLUS = ceux dont 'mRS 3 months' est renseigné dans outcome.csv.
  2. Lire (sans extraire) les 13 .zip NeMo et lister, par patient, les .pkl
     ChaCo essentiels :
        - chacoconn ...mean.pkl  -> matrice 143x143 (le graphe : edges)
        - chacovol  ...mean.pkl  -> vecteur 143     (features de noeuds)
  3. Recouper INCLUS <-> ChaCo et sortir :
        OK        : inclus + 2 .pkl présents
        INCOMPLET : inclus mais au moins un .pkl manquant
        SANS_NEMO : inclus mais aucun ChaCo trouve
        ORPHELIN  : ChaCo present mais patient non inclus (mRS manquant)
  4. Charger un .pkl temoin et verifier les shapes (143,143) et (143,).
  5. Ecrire inventaire_nemo_qc.csv (1 ligne / patient).

Aucune extraction des zip n'est faite : lecture en flux via zipfile.
--------------------------------------------------------------------------------
ADAPTER UNIQUEMENT LES 2 CHEMINS CI-DESSOUS, puis lancer le script.
"""

import os
import re
import csv
import zipfile
import pickle
import io

# ============================ A ADAPTER ========================================
# Dossier contenant les sous-dossiers patients sub-stroke0001/, sub-stroke0002/...
MNI_DIR = "/content/drive/MyDrive/isles_prp/MNI"

# Dossier contenant les 13 .zip NeMo (batch_01_..._.zip ... batch_13_..._.zip)
ZIPS_DIR = "/content/drive/MyDrive/isles_prp/MNI/nemo_results"

# Nom exact de la colonne mRS cible dans outcome.csv
MRS_COLUMN = "mRS 3 months"
# ===============================================================================

# Fichier de sortie (ecrit a cote des zips)
OUT_CSV = os.path.join(ZIPS_DIR, "inventaire_nemo_qc.csv")

# Regex pour extraire l'identifiant patient (sub-strokeXXXX) depuis un nom de fichier
PID_RE = re.compile(r"(sub-stroke\d+)", re.IGNORECASE)

# Suffixes des 2 .pkl essentiels (on cherche '...mean.pkl')
CONN_TOKEN = "chacoconn"   # matrice 143x143
VOL_TOKEN  = "chacovol"    # vecteur 143
MEAN_TOKEN = "mean.pkl"    # on prend la moyenne, pas le stdev


# --------------------------------------------------------------------------- #
# 1) Construire la liste des INCLUS (mRS renseigne)                            #
# --------------------------------------------------------------------------- #
def lire_mrs(outcome_path):
    """Retourne la valeur mRS 3 months (str) ou None si absente/vide."""
    try:
        with open(outcome_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # normaliser les en-tetes (espaces parasites eventuels)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                val = (row.get(MRS_COLUMN) or "").strip()
                return val if val != "" else None
    except FileNotFoundError:
        return None
    return None


def construire_inclus(mni_dir):
    """Parcourt MNI/sub-stroke* et separe inclus / exclus selon le mRS."""
    inclus, exclus = {}, {}
    if not os.path.isdir(mni_dir):
        raise SystemExit(f"[ERREUR] Dossier MNI introuvable : {mni_dir}")

    for name in sorted(os.listdir(mni_dir)):
        d = os.path.join(mni_dir, name)
        if not os.path.isdir(d):
            continue
        m = PID_RE.search(name)
        if not m:
            continue
        pid = m.group(1).lower()

        # chercher le outcome.csv (nom: sub-strokeXXXX_ses-01_outcome.csv)
        outcome = None
        for fn in os.listdir(d):
            if fn.lower().endswith("outcome.csv"):
                outcome = os.path.join(d, fn)
                break

        mrs = lire_mrs(outcome) if outcome else None
        if mrs is None:
            exclus[pid] = "mRS manquant" if outcome else "outcome.csv absent"
        else:
            inclus[pid] = mrs
    return inclus, exclus


# --------------------------------------------------------------------------- #
# 2) Indexer les ChaCo presents dans les .zip (sans extraire)                  #
# --------------------------------------------------------------------------- #
def indexer_chaco(zips_dir):
    """
    Retourne :
      chaco[pid] = {'conn': (zip, membre) | None, 'vol': (zip, membre) | None}
      zips_vus   = liste des zip parcourus
    """
    chaco = {}
    zips_vus = []
    if not os.path.isdir(zips_dir):
        raise SystemExit(f"[ERREUR] Dossier zips introuvable : {zips_dir}")

    zip_files = sorted(fn for fn in os.listdir(zips_dir)
                       if fn.lower().endswith(".zip"))
    if not zip_files:
        raise SystemExit(f"[ERREUR] Aucun .zip dans : {zips_dir}")

    for zf in zip_files:
        zpath = os.path.join(zips_dir, zf)
        try:
            with zipfile.ZipFile(zpath) as z:
                zips_vus.append(zf)
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
                        slot["conn"] = (zf, member)
                    elif VOL_TOKEN in base and slot["vol"] is None:
                        slot["vol"] = (zf, member)
        except zipfile.BadZipFile:
            print(f"[ALERTE] ZIP corrompu, ignore : {zf}")
    return chaco, zips_vus


# --------------------------------------------------------------------------- #
# 3) Charger un .pkl temoin depuis un zip et verifier la shape                 #
# --------------------------------------------------------------------------- #
def shape_de(zips_dir, ref):
    """ref = (nom_zip, membre). Retourne la shape ou un message."""
    if ref is None:
        return "—"
    zf, member = ref
    try:
        with zipfile.ZipFile(os.path.join(zips_dir, zf)) as z:
            with z.open(member) as fh:
                obj = pickle.load(io.BytesIO(fh.read()))
        if hasattr(obj, "shape"):
            return str(obj.shape)
        try:
            return f"len={len(obj)}"
        except TypeError:
            return type(obj).__name__
    except Exception as e:
        return f"err:{type(e).__name__}"


# --------------------------------------------------------------------------- #
# MAIN                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print(" INVENTAIRE QC NeMo / ChaCo  <->  cohorte mRS")
    print("=" * 70)

    inclus, exclus = construire_inclus(MNI_DIR)
    chaco, zips_vus = indexer_chaco(ZIPS_DIR)

    print(f"\nZIP parcourus           : {len(zips_vus)}")
    print(f"Patients avec mRS (inclus) : {len(inclus)}")
    print(f"Patients exclus (mRS vide) : {len(exclus)}")
    print(f"Patients avec ChaCo trouve : {len(chaco)}")

    # Categorisation
    ok, incomplet, sans_nemo, orphelin = [], [], [], []

    for pid in sorted(inclus):
        c = chaco.get(pid)
        if c is None:
            sans_nemo.append(pid)
        elif c["conn"] and c["vol"]:
            ok.append(pid)
        else:
            manque = []
            if not c["conn"]:
                manque.append("conn")
            if not c["vol"]:
                manque.append("vol")
            incomplet.append((pid, "+".join(manque)))

    for pid in sorted(chaco):
        if pid not in inclus:
            orphelin.append(pid)

    # Rapport console
    print("\n" + "-" * 70)
    print(f"  OK (inclus + 2 pkl)        : {len(ok)}")
    print(f"  INCOMPLET (1 pkl manquant) : {len(incomplet)}")
    print(f"  SANS_NEMO (inclus, 0 ChaCo): {len(sans_nemo)}")
    print(f"  ORPHELIN (ChaCo sans mRS)  : {len(orphelin)}")
    print("-" * 70)

    if incomplet:
        print("\n[INCOMPLET]")
        for pid, m in incomplet:
            print(f"   {pid}  (manque: {m})")
    if sans_nemo:
        print("\n[SANS_NEMO]  (inclus mais aucun ChaCo dans les zip)")
        for pid in sans_nemo:
            print(f"   {pid}")
    if orphelin:
        print("\n[ORPHELIN]  (ChaCo present mais mRS non renseigne -> exclu)")
        for pid in orphelin:
            print(f"   {pid}  ({exclus.get(pid, 'non inclus')})")

    # Verification des shapes sur un patient OK temoin
    print("\n" + "-" * 70)
    if ok:
        temoin = ok[0]
        c = chaco[temoin]
        print(f"Verification shapes sur temoin : {temoin}")
        print(f"   chacoconn (attendu (143,143)) : {shape_de(ZIPS_DIR, c['conn'])}")
        print(f"   chacovol  (attendu (143,))    : {shape_de(ZIPS_DIR, c['vol'])}")
    else:
        print("Aucun patient OK : impossible de verifier les shapes.")
    print("-" * 70)

    # Ecriture CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "mrs_3months", "statut",
                    "conn_pkl", "vol_pkl", "zip_source"])
        for pid in sorted(set(list(inclus) + list(chaco))):
            mrs = inclus.get(pid, "")
            c = chaco.get(pid, {"conn": None, "vol": None})
            if pid in inclus and c["conn"] and c["vol"]:
                statut = "OK"
            elif pid in inclus and (c["conn"] or c["vol"]):
                statut = "INCOMPLET"
            elif pid in inclus:
                statut = "SANS_NEMO"
            else:
                statut = "ORPHELIN"
            zip_src = (c["conn"] or c["vol"] or ("", ""))[0]
            w.writerow([pid, mrs, statut,
                        "oui" if c["conn"] else "non",
                        "oui" if c["vol"] else "non",
                        zip_src])

    print(f"\nCSV ecrit : {OUT_CSV}")
    print("\nRESUME :")
    print(f"   Cohorte finale exploitable (OK) = {len(ok)} patients")
    if len(ok) == 129:
        print("   -> Conforme au verrou n=129. ✅")
    else:
        print(f"   -> Attendu 129, obtenu {len(ok)}. A investiguer (voir listes ci-dessus).")


if __name__ == "__main__":
    main()
