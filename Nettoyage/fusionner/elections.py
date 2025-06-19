# import os
# import pandas as pd

# # === CONFIGURATION DES CHEMINS ===
# # Chemins des fichiers sources (ajustez selon vos chemins)
# fichier_2022 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_tour-1/DATA_Nettoyer_t1_2022.xlsx"
# fichier_2017 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_tour-1/DATA_Nettoyer_t1_2017.xlsx"

# # Dossier de sortie pour le fichier fusionné
# dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Fusionnees"

# print("=== FUSION DES DATASETS 2022 ET 2017 ===")
# print("=" * 50)

# # === 1. CHARGER LES DEUX DATASETS ===
# try:
#     print("📂 Chargement du dataset 2022...")
#     df_2022 = pd.read_excel(fichier_2022)
#     print(f"   ✅ Dataset 2022 chargé : {len(df_2022)} lignes, {len(df_2022.columns)} colonnes")
    
#     print("📂 Chargement du dataset 2017...")
#     df_2017 = pd.read_excel(fichier_2017)
#     print(f"   ✅ Dataset 2017 chargé : {len(df_2017)} lignes, {len(df_2017.columns)} colonnes")
    
# except FileNotFoundError as e:
#     print(f"❌ Erreur : Fichier non trouvé - {e}")
#     print("Vérifiez les chemins des fichiers dans le code.")
#     exit()

# # === 2. VÉRIFICATION DE LA COMPATIBILITÉ ===
# print("\n🔍 Vérification de la compatibilité des datasets...")

# # Vérifier si les colonnes sont identiques
# colonnes_2022 = set(df_2022.columns)
# colonnes_2017 = set(df_2017.columns)

# if colonnes_2022 == colonnes_2017:
#     print("   ✅ Les colonnes sont identiques dans les deux datasets")
# else:
#     print("   ⚠️  Différences détectées dans les colonnes :")
#     colonnes_uniquement_2022 = colonnes_2022 - colonnes_2017
#     colonnes_uniquement_2017 = colonnes_2017 - colonnes_2022
    
#     if colonnes_uniquement_2022:
#         print(f"      Uniquement dans 2022 : {list(colonnes_uniquement_2022)}")
#     if colonnes_uniquement_2017:
#         print(f"      Uniquement dans 2017 : {list(colonnes_uniquement_2017)}")

# # Afficher les colonnes communes
# colonnes_communes = list(colonnes_2022.intersection(colonnes_2017))
# print(f"   📋 Colonnes communes : {len(colonnes_communes)}")

# # === 3. AFFICHAGE DES INFORMATIONS AVANT FUSION ===
# print("\n📊 Informations avant fusion :")
# print(f"   Dataset 2022 : {len(df_2022)} communes")
# print(f"   Dataset 2017 : {len(df_2017)} communes")

# # === 4. FUSION DES DATASETS ===
# print("\n🔗 Fusion des datasets en cours...")

# # Méthode 1 : Si les colonnes sont identiques
# if colonnes_2022 == colonnes_2017:
#     # Fusion simple : 2022 en haut, 2017 en bas
#     df_fusionne = pd.concat([df_2022, df_2017], ignore_index=True)
# else:
#     # Méthode 2 : Si les colonnes diffèrent, on utilise les colonnes communes
#     print("   ⚙️  Utilisation des colonnes communes pour la fusion...")
#     df_2022_filtre = df_2022[colonnes_communes]
#     df_2017_filtre = df_2017[colonnes_communes]
#     df_fusionne = pd.concat([df_2022_filtre, df_2017_filtre], ignore_index=True)

# print(f"   ✅ Fusion terminée : {len(df_fusionne)} lignes au total")

# # === 5. VÉRIFICATION DE LA FUSION ===
# print("\n🔍 Vérification de la fusion :")

# # Compter les lignes par année
# repartition_annees = df_fusionne['Année'].value_counts().sort_index()
# print("   📊 Répartition par année :")
# for annee, count in repartition_annees.items():
#     print(f"      {annee} : {count} communes")

# # Vérifier s'il y a des doublons d'ID Commune pour la même année
# doublons_par_annee = df_fusionne.groupby('Année')['ID Commune'].apply(lambda x: x.duplicated().sum())
# print("\n   🔍 Vérification des doublons par année :")
# for annee, nb_doublons in doublons_par_annee.items():
#     if nb_doublons > 0:
#         print(f"      ⚠️  {annee} : {nb_doublons} doublons détectés")
#     else:
#         print(f"      ✅ {annee} : Aucun doublon")

# # === 6. STATISTIQUES DESCRIPTIVES ===
# print("\n📈 Statistiques du dataset fusionné :")
# print(f"   Nombre total de lignes : {len(df_fusionne)}")
# print(f"   Nombre de colonnes : {len(df_fusionne.columns)}")
# print(f"   Colonnes : {list(df_fusionne.columns)}")

# # Vérifier les valeurs manquantes
# print("\n   📋 Valeurs manquantes par colonne :")
# valeurs_manquantes = df_fusionne.isnull().sum()
# for col, nb_nan in valeurs_manquantes.items():
#     if nb_nan > 0:
#         print(f"      {col} : {nb_nan} valeurs manquantes")

# # === 7. RÉORGANISATION FINALE ===
# print("\n🔄 Réorganisation finale...")

# # S'assurer que les données sont triées : 2022 d'abord, puis 2017
# df_fusionne = df_fusionne.sort_values(['Année', 'ID Commune'], ascending=[False, True])
# df_fusionne = df_fusionne.reset_index(drop=True)

# print("   ✅ Données triées : 2022 en premier, puis 2017")

# # === 8. SAUVEGARDE ===
# def sauvegarder_fichiers(df, nom_base, dossier):
#     """Sauvegarde le dataframe en Excel et CSV"""
#     os.makedirs(dossier, exist_ok=True)
    
#     # Sauvegarde Excel
#     chemin_excel = os.path.join(dossier, f"{nom_base}.xlsx")
#     try:
#         df.to_excel(chemin_excel, index=False, engine='openpyxl')
#         print(f"   ✅ Fichier Excel sauvegardé : {chemin_excel}")
#     except Exception as e:
#         print(f"   ❌ Erreur Excel : {e}")
    
#     # Sauvegarde CSV
#     chemin_csv = os.path.join(dossier, f"{nom_base}.csv")
#     try:
#         df.to_csv(chemin_csv, index=False, encoding='utf-8-sig', sep=';')
#         print(f"   ✅ Fichier CSV sauvegardé : {chemin_csv}")
#     except Exception as e:
#         print(f"   ❌ Erreur CSV : {e}")
    
#     return chemin_excel, chemin_csv

# print("\n💾 Sauvegarde des fichiers fusionnés...")
# nom_fichier_fusionne = "DATA_Fusionnees_2022_2017"
# chemin_excel, chemin_csv = sauvegarder_fichiers(df_fusionne, nom_fichier_fusionne, dossier_sortie)

# # === 9. RAPPORT FINAL ===
# print("\n" + "=" * 50)
# print("📋 RAPPORT FINAL DE FUSION")
# print("=" * 50)
# print(f"✅ Fusion réussie !")
# print(f"📊 Dataset final : {len(df_fusionne)} lignes")
# print(f"🗓️  Années : {sorted(df_fusionne['Année'].unique(), reverse=True)}")
# print(f"🏛️  Communes uniques : {df_fusionne['ID Commune'].nunique()}")
# print(f"📁 Fichiers générés :")
# print(f"   📄 Excel : {chemin_excel}")
# print(f"   📄 CSV   : {chemin_csv}")

# # Aperçu des premières lignes
# print(f"\n👀 Aperçu des 5 premières lignes :")
# print(df_fusionne[['ID Commune', 'Année', 'Nom Complet Élu', 'Parti Politique Élu']].head())

# print("\n🎉 Fusion terminée avec succès !")
# print("=" * 50)
import os
import pandas as pd

# === CONFIGURATION DES CHEMINS ===
fichier_2022 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/Data_elections/DATA_Nettoyer_t1_2022.xlsx"
fichier_2017 = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_elections/DATA_Nettoyer_t1_2017.xlsx"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Fusionnees"

print("=== FUSION DES DATASETS PAR ID COMMUNE ===")
print("=" * 50)

# === CHARGEMENT ===
try:
    print("📂 Chargement des datasets...")
    df_2022 = pd.read_excel(fichier_2022)
    df_2017 = pd.read_excel(fichier_2017)
except FileNotFoundError as e:
    print(f"❌ Fichier non trouvé : {e}")
    exit()

# === VERIFICATION COLONNES ===
colonnes_2022 = set(df_2022.columns)
colonnes_2017 = set(df_2017.columns)
colonnes_communes = list(colonnes_2022 & colonnes_2017)

if colonnes_2022 != colonnes_2017:
    print("⚠️ Colonnes différentes, utilisation des colonnes communes.")

df_2022 = df_2022[colonnes_communes].copy()
df_2017 = df_2017[colonnes_communes].copy()

# S'assurer que la colonne "Année" existe
if "Année" not in df_2022.columns:
    df_2022["Année"] = 2022
if "Année" not in df_2017.columns:
    df_2017["Année"] = 2017

# === FUSION ALTERNÉE PAR ID COMMUNE ===
print("🔗 Fusion alternée par ID Commune...")

# Fusion sur ID Commune
df_2022_sorted = df_2022.sort_values("ID Commune")
df_2017_sorted = df_2017.sort_values("ID Commune")

# Faire un merge outer pour capturer toutes les communes des deux années
ids_communes = pd.Series(sorted(set(df_2022["ID Commune"]) | set(df_2017["ID Commune"])))

# Créer une liste de lignes fusionnées
fusion_lignes = []
for id_commune in ids_communes:
    ligne_2022 = df_2022_sorted[df_2022_sorted["ID Commune"] == id_commune]
    ligne_2017 = df_2017_sorted[df_2017_sorted["ID Commune"] == id_commune]
    if not ligne_2022.empty:
        fusion_lignes.append(ligne_2022)
    if not ligne_2017.empty:
        fusion_lignes.append(ligne_2017)

df_fusionne = pd.concat(fusion_lignes, ignore_index=True)

# === TRI FINAL ===
df_fusionne = df_fusionne.sort_values(["ID Commune", "Année"], ascending=[True, False]).reset_index(drop=True)
# === RÉORGANISATION DES COLONNES DANS L’ORDRE SOUHAITÉ ===
ordre_colonnes = [
    "ID Commune", "Année", "Inscrits", "% Voix/Ins Élu", "% Voix/Exp Élu", "Abstentions",
    "Sexe Élu", "Nom Complet Élu", "Voix Élu", "Parti Politique Élu", "Orientation Politique", "Score Orientation (0 à 4)"
]

# Filtrer uniquement les colonnes présentes dans le DataFrame
colonnes_finales = [col for col in ordre_colonnes if col in df_fusionne.columns]
df_fusionne = df_fusionne[colonnes_finales]

# === SAUVEGARDE ===
def sauvegarder(df, nom_base, dossier):
    os.makedirs(dossier, exist_ok=True)
    chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
    chemin_csv = os.path.join(dossier, nom_base + ".csv")

    df.to_excel(chemin_xlsx, index=False, engine='openpyxl')
    df.to_csv(chemin_csv, index=False, sep=';', encoding='utf-8-sig')
    
    return chemin_xlsx, chemin_csv

print("\n💾 Sauvegarde du fichier fusionné...")
nom_fichier = "DATA_Nettoyer_Elections"
chemin_excel, chemin_csv = sauvegarder(df_fusionne, nom_fichier, dossier_sortie)

# === RAPPORT FINAL ===
print("\n📋 RAPPORT FINAL")
print("=" * 50)
print(f"✅ Fusion alternée réussie")
print(f"📄 Excel : {chemin_excel}")
print(f"📄 CSV   : {chemin_csv}")
print(f"📊 Lignes : {len(df_fusionne)}")
print(f"🗓️  Années incluses : {df_fusionne['Année'].unique()}")
print(f"🏛️  Communes uniques : {df_fusionne['ID Commune'].nunique()}")
print("=" * 50)
