import os
import pandas as pd

# === CONFIGURATION DES CHEMINS ===
fichier_2021 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/revenu_2021.csv"
fichier_2016 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/revenu_2016.csv"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Fusionnees"

print("=== FUSION DES DATASETS 2021 ET 2016 PAR ID COMMUNE ===")
print("=" * 50)

# === 1. CHARGEMENT ===
try:
    print("📂 Chargement des fichiers...")
    df_2021 = pd.read_csv(fichier_2021, sep=',')
    df_2016 = pd.read_csv(fichier_2016, sep=',')
except FileNotFoundError as e:
    print(f"❌ Erreur : {e}")
    exit()

# === 2. VÉRIFICATION DES COLONNES COMMUNES ===
colonnes_2021 = set(df_2021.columns)
colonnes_2016 = set(df_2016.columns)
colonnes_communes = list(colonnes_2021 & colonnes_2016)

if colonnes_2021 != colonnes_2016:
    print("⚠️ Colonnes différentes détectées. Fusion basée sur les colonnes communes.")

df_2021 = df_2021[colonnes_communes].copy()
df_2016 = df_2016[colonnes_communes].copy()

# === 3. AJOUT DES ANNÉES ===
df_2021["Année"] = 2021
df_2016["Année"] = 2016

# === 4. FORMATTAGE DES ID COMMUNE ===
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
nom_fichier = "DATA_Fusionnees_revenu_2021_2016"
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
