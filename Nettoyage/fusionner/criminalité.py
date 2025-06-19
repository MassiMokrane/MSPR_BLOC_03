import os
import pandas as pd

# === CONFIGURATION DES CHEMINS ===
fichier_2021 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/criminalite_2021.csv"
fichier_2016 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/criminalite_2016.csv"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Fusionnees"

print("=== FUSION DES DONNÉES CRIMINALITÉ 2021 & 2016 (ID + Année + nb_faits) ===")
print("=" * 60)

# === 1. CHARGEMENT ===
try:
    print("📂 Chargement des fichiers...")
    df_2021 = pd.read_csv(fichier_2021, sep=',')
    df_2016 = pd.read_csv(fichier_2016, sep=',')
except FileNotFoundError as e:
    print(f"❌ Erreur : {e}")
    exit()

# === 2. SÉLECTION DES COLONNES UTILES UNIQUEMENT ===
colonnes_utiles = ["ID Commune", "nb_faits"]

df_2021 = df_2021[colonnes_utiles].copy()
df_2016 = df_2016[colonnes_utiles].copy()

# === 3. AJOUT DE LA COLONNE ANNÉE ===
df_2021["Année"] = 2021
df_2016["Année"] = 2016

# === 4. FORMATTAGE DE L’ID COMMUNE ===
df_2021["ID Commune"] = df_2021["ID Commune"].astype(str).str.zfill(5)
df_2016["ID Commune"] = df_2016["ID Commune"].astype(str).str.zfill(5)

# === 5. FUSION ALTERNÉE PAR ID COMMUNE ===
print("🔁 Fusion alternée par ID Commune...")

ids_communes = pd.Series(sorted(set(df_2021["ID Commune"]) | set(df_2016["ID Commune"])))

fusion_lignes = []
for idc in ids_communes:
    ligne_2021 = df_2021[df_2021["ID Commune"] == idc]
    ligne_2016 = df_2016[df_2016["ID Commune"] == idc]
    if not ligne_2021.empty:
        fusion_lignes.append(ligne_2021)
    if not ligne_2016.empty:
        fusion_lignes.append(ligne_2016)

df_fusionne = pd.concat(fusion_lignes, ignore_index=True)
df_fusionne = df_fusionne.sort_values(["ID Commune", "Année"], ascending=[True, False]).reset_index(drop=True)

# === 6. SAUVEGARDE DES FICHIERS ===
def sauvegarder(df, nom_base, dossier):
    os.makedirs(dossier, exist_ok=True)
    chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
    chemin_csv = os.path.join(dossier, nom_base + ".csv")

    df.to_excel(chemin_xlsx, index=False, engine='openpyxl')
    df.to_csv(chemin_csv, index=False, sep=';', encoding='utf-8-sig')

    return chemin_xlsx, chemin_csv

print("\n💾 Sauvegarde des fichiers...")
nom_fichier = "DATA_Criminalite_2021_2016"
chemin_excel, chemin_csv = sauvegarder(df_fusionne, nom_fichier, dossier_sortie)

# === 7. RAPPORT FINAL ===
print("\n📋 RAPPORT FINAL")
print("=" * 50)
print(f"✅ Fusion réussie avec alternance 2021 > 2016")
print(f"📁 Excel : {chemin_excel}")
print(f"📁 CSV   : {chemin_csv}")
print(f"📊 Total lignes : {len(df_fusionne)}")
print(f"🗓️  Années présentes : {df_fusionne['Année'].unique()}")
print(f"🏛️  Communes uniques : {df_fusionne['ID Commune'].nunique()}")
print("=" * 50)
