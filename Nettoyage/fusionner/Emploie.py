import os
import pandas as pd

# === CONFIGURATION DES CHEMINS ===
fichier_2022 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/emploi_nettoye_2022.csv"
fichier_2017 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_nettoyer/emploi_nettoye_2017.csv"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Fusionnees"

print("=== FUSION DES DONNÉES EMPLOI 2022 & 2017 (ID Commune + Année + indicateurs) ===")
print("=" * 60)

def nettoyer_colonnes(df):
    """
    Nettoie les colonnes numériques en :
    - remplaçant les virgules par des points (si besoin)
    - convertissant en int toutes les colonnes sauf '% Chômage'
    - convertissant '% Chômage' en float
    """
    for col in df.columns:
        if col == "ID Commune" or col == "Année":
            continue
        # Remplacement virgule par point (au cas où)
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        if col == "% Chômage":
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            # Conversion en float d'abord (pour les valeurs décimales qui peuvent apparaître),
            # puis arrondi et conversion en int
            df[col] = pd.to_numeric(df[col], errors='coerce').round(0).astype('Int64')
    return df

# === 1. CHARGEMENT ===
try:
    print("📂 Chargement des fichiers...")
    df_2022 = pd.read_csv(fichier_2022, sep=';')
    df_2017 = pd.read_csv(fichier_2017, sep=';')
except FileNotFoundError as e:
    print(f"❌ Erreur : {e}")
    exit()

# === 2. SÉLECTION DES COLONNES UTILES UNIQUEMENT ===
colonnes_utiles = ["ID Commune", "Population Active", "Chômeurs", "Emplois", "% Chômage"]

df_2022 = df_2022[colonnes_utiles].copy()
df_2017 = df_2017[colonnes_utiles].copy()

# === 3. NETTOYAGE DES VALEURS ===
df_2022 = nettoyer_colonnes(df_2022)
df_2017 = nettoyer_colonnes(df_2017)

# === 4. AJOUT DE LA COLONNE ANNÉE ===
df_2022["Année"] = 2022
df_2017["Année"] = 2017

# === 5. FORMATTAGE DE L’ID COMMUNE ===
df_2022["ID Commune"] = df_2022["ID Commune"].astype(str).str.zfill(5)
df_2017["ID Commune"] = df_2017["ID Commune"].astype(str).str.zfill(5)

# === 6. FUSION ALTERNÉE PAR ID COMMUNE ===
print("🔁 Fusion alternée par ID Commune...")

ids_communes = pd.Series(sorted(set(df_2022["ID Commune"]) | set(df_2017["ID Commune"])))

fusion_lignes = []
for idc in ids_communes:
    ligne_2022 = df_2022[df_2022["ID Commune"] == idc]
    ligne_2017 = df_2017[df_2017["ID Commune"] == idc]
    if not ligne_2022.empty:
        fusion_lignes.append(ligne_2022)
    if not ligne_2017.empty:
        fusion_lignes.append(ligne_2017)

df_fusionne = pd.concat(fusion_lignes, ignore_index=True)
df_fusionne = df_fusionne.sort_values(["ID Commune", "Année"], ascending=[True, False]).reset_index(drop=True)

# === 7. SAUVEGARDE DES FICHIERS ===
def sauvegarder(df, nom_base, dossier):
    os.makedirs(dossier, exist_ok=True)
    chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
    chemin_csv = os.path.join(dossier, nom_base + ".csv")

    df.to_excel(chemin_xlsx, index=False, engine='openpyxl')
    df.to_csv(chemin_csv, index=False, sep=';', encoding='utf-8-sig')

    return chemin_xlsx, chemin_csv

print("\n💾 Sauvegarde des fichiers...")
nom_fichier = "DATA_Emploi_2022_2017"
chemin_excel, chemin_csv = sauvegarder(df_fusionne, nom_fichier, dossier_sortie)

# === 8. RAPPORT FINAL ===
print("\n📋 RAPPORT FINAL")
print("=" * 50)
print(f"✅ Fusion réussie avec alternance 2022 > 2017")
print(f"📁 Excel : {chemin_excel}")
print(f"📁 CSV   : {chemin_csv}")
print(f"📊 Total lignes : {len(df_fusionne)}")
print(f"🗓️  Années présentes : {df_fusionne['Année'].unique()}")
print(f"🏛️  Communes uniques : {df_fusionne['ID Commune'].nunique()}")
print("=" * 50)
