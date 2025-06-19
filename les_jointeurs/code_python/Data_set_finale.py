import pandas as pd
import os
import numpy as np

# === Chemins ===
fichier_elections = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Nettoyer_Elections.csv"
fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Population_finale.csv"
fichier_criminaliter = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Criminalite_2021_2016.csv"
fichier_emploi = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Emploi_2022_2017.csv"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/"
os.makedirs(dossier_sortie, exist_ok=True)

print("🔄 Lecture des données...")
# === Lecture des données ===
df_elec_full = pd.read_csv(fichier_elections, sep=';')
df_pop_full = pd.read_csv(fichier_population, sep=',')
df_crime_full = pd.read_csv(fichier_criminaliter, sep=';')
df_emploi_full = pd.read_csv(fichier_emploi, sep=';')

# === Sélection des colonnes spécifiées avec noms originaux ===
print("📋 Sélection des colonnes...")

# Elections: Colonnes mises à jour selon le nouveau format
colonnes_elec = ['ID Commune', 'Année', 'Inscrits', 'Abstentions', 
                 'Score Orientation (0 à 4)']
df_elec = df_elec_full[colonnes_elec].copy()

# Population: Colonnes mises à jour selon le nouveau dataset
colonnes_pop = [
    'ID_Commune', 'Annee', 'Population_Totale',
    'Pct_Jeunes', 'Pct_Seniors', 
    'Pct_Sans_Activite', 'Pct_Etrangere'
    #   'Taille_Commune'
]
df_pop = df_pop_full[colonnes_pop].copy()

# Criminalité: Colonnes selon le nouveau format
colonnes_crime = ['ID Commune', 'Année', 'nb_crimes']
df_crime = df_crime_full[colonnes_crime].copy()

# Emploi: Colonnes selon le nouveau format
colonnes_emploi = ['ID Commune', 'Année', 'Population Active',
                #    'Emplois'
                 '% Chômage']
df_emploi = df_emploi_full[colonnes_emploi].copy()

# === CONVERSION ET NETTOYAGE DES TYPES DE DONNÉES ===
print("🔧 Conversion des types de données...")

# CORRECTION: Nettoyer et convertir orientation_politique en int
print("   - Nettoyage de la colonne orientation politique...")
# Supprimer les espaces et caractères non désirés
df_elec['Score Orientation (0 à 4)'] = df_elec['Score Orientation (0 à 4)'].astype(str).str.strip()
# Remplacer les valeurs vides ou 'nan' par NaN
df_elec.loc[df_elec['Score Orientation (0 à 4)'].isin(['', 'nan', 'NaN', 'null']), 'Score Orientation (0 à 4)'] = np.nan
# Convertir en float d'abord, puis en int (pour gérer les NaN)
df_elec['Score Orientation (0 à 4)'] = pd.to_numeric(df_elec['Score Orientation (0 à 4)'], errors='coerce')

# === RENOMMAGE PROFESSIONNEL DES COLONNES ===
print("🏷️ Renommage professionnel des colonnes...")

# Elections - Renommage professionnel
df_elec.rename(columns={
    'ID Commune': 'commune_id',
    'Année': 'annee',
    'Inscrits': 'nb_inscrits',
    'Abstentions': 'nb_abstentions',
    'Score Orientation (0 à 4)': 'orientation_politique'
}, inplace=True)

# CORRECTION: Convertir orientation_politique en int après renommage
print("   - Conversion orientation_politique en int...")
df_elec['orientation_politique'] = df_elec['orientation_politique'].astype('Int64')  # Int64 permet les NaN

# Population - Renommage professionnel
df_pop.rename(columns={
    'ID_Commune': 'commune_id',
    'Annee': 'annee',
    'Population_Totale':'Population_Totale',
    'Pct_Jeunes': 'pct_population_jeune',
    'Pct_Seniors': 'pct_population_senior',
    'Pct_Sans_Activite': 'pct_population_sans_activite',
    'Pct_Etrangere': 'pct_population_etrangere',
    # 'Taille_Commune': 'taille_commune_categorie'
}, inplace=True)

# Criminalité - Renommage professionnel
df_crime.rename(columns={
    'ID Commune': 'commune_id',
    'Année': 'annee',
    'nb_crimes': 'nb_crimes'
}, inplace=True)

# Emploi - Renommage professionnel
df_emploi.rename(columns={
    'ID Commune': 'commune_id',
    'Année': 'annee',
    'Population Active': 'nb_population_active',
    # 'Emplois': 'nb_emplois_total',
    '% Chômage': 'taux_chomage_pct'
}, inplace=True)

print(f"📊 Données chargées:")
print(f"   - Elections: {len(df_elec)} lignes, {len(df_elec.columns)} colonnes")
print(f"   - Population: {len(df_pop)} lignes, {len(df_pop.columns)} colonnes")
print(f"   - Criminalité: {len(df_crime)} lignes, {len(df_crime.columns)} colonnes")
print(f"   - Emploi: {len(df_emploi)} lignes, {len(df_emploi.columns)} colonnes")

# CORRECTION: Vérifier le type de la colonne orientation_politique
print(f"   - Type orientation_politique: {df_elec['orientation_politique'].dtype}")

# === Nettoyage des IDs ===
for df in [df_elec, df_pop, df_crime, df_emploi]:
    df['commune_id'] = df['commune_id'].astype(str).str.zfill(5)

# === Filtrer jusqu'à ID Commune 95690 inclus ===
df_elec = df_elec[df_elec['commune_id'] <= '95690']
df_pop = df_pop[df_pop['commune_id'] <= '95690']
df_crime = df_crime[df_crime['commune_id'] <= '95690']
df_emploi = df_emploi[df_emploi['commune_id'] <= '95690']

# === Convertir les années ===
df_elec['annee'] = df_elec['annee'].astype(int)
df_pop['annee'] = df_pop['annee'].astype(int)
df_crime['annee'] = df_crime['annee'].astype(int)
df_emploi['annee'] = df_emploi['annee'].astype(int)

print("\n🔍 Vérification de la complétude des données par commune...")

# === Fonction pour vérifier la complétude d'un DataFrame ===
def verifier_completude(df, nom_dataset):
    """Vérifie qu'un DataFrame n'a pas de valeurs manquantes"""
    lignes_avec_na = df.isnull().any(axis=1).sum()
    print(f"   - {nom_dataset}: {lignes_avec_na} lignes avec valeurs manquantes")
    return df.dropna()

# === Nettoyage des valeurs manquantes ===
print("\n🧹 Suppression des lignes avec valeurs manquantes...")
df_elec_clean = verifier_completude(df_elec, "Elections")
df_pop_clean = verifier_completude(df_pop, "Population")
df_crime_clean = verifier_completude(df_crime, "Criminalité")
df_emploi_clean = verifier_completude(df_emploi, "Emploi")

print(f"\n📋 Données après nettoyage:")
print(f"   - Elections: {len(df_elec_clean)} lignes")
print(f"   - Population: {len(df_pop_clean)} lignes")
print(f"   - Criminalité: {len(df_crime_clean)} lignes")
print(f"   - Emploi: {len(df_emploi_clean)} lignes")

# === Identifier les communes avec données complètes pour chaque période ===

def trouver_communes_completes(df_elec, df_pop, df_crime, df_emploi, annee_elec, annee_pop, annee_crime, annee_emploi):
    """Trouve les communes ayant des données complètes pour toutes les sources pour une période donnée"""
    
    communes_elec = set(df_elec[df_elec['annee'] == annee_elec]['commune_id'])
    communes_pop = set(df_pop[df_pop['annee'] == annee_pop]['commune_id'])
    communes_crime = set(df_crime[df_crime['annee'] == annee_crime]['commune_id'])
    communes_emploi = set(df_emploi[df_emploi['annee'] == annee_emploi]['commune_id'])
    
    # Intersection de toutes les communes (présentes dans tous les datasets)
    communes_completes = communes_elec.intersection(communes_pop).intersection(communes_crime).intersection(communes_emploi)
    
    print(f"\n📈 Période {annee_elec} (pop:{annee_pop}, crime:{annee_crime}, emploi:{annee_emploi}):")
    print(f"   - Elections: {len(communes_elec)} communes")
    print(f"   - Population: {len(communes_pop)} communes")
    print(f"   - Criminalité: {len(communes_crime)} communes")
    print(f"   - Emploi: {len(communes_emploi)} communes")
    print(f"   - ✅ Communes complètes: {len(communes_completes)} communes")
    
    return communes_completes

# === Trouver les communes complètes pour chaque période ===
print("\n🔎 Identification des communes avec données complètes...")

# Vérifier les années disponibles dans chaque dataset
print("📅 Années disponibles dans chaque dataset:")
print(f"   - Elections: {sorted(df_elec_clean['annee'].unique())}")
print(f"   - Population: {sorted(df_pop_clean['annee'].unique())}")
print(f"   - Criminalité: {sorted(df_crime_clean['annee'].unique())}")
print(f"   - Emploi: {sorted(df_emploi_clean['annee'].unique())}")

# Adapter les périodes selon les données disponibles
# Période récente (2022) - si disponible
if 2022 in df_elec_clean['annee'].values:
    communes_periode_2022 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                       2022, 2021, 2021, 2022)
else:
    communes_periode_2022 = set()

# Si pas de 2022, essayer avec 2021
if 2021 in df_elec_clean['annee'].values and len(communes_periode_2022) == 0:
    communes_periode_2021 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                       2021, 2021, 2021, 2022)
else:
    communes_periode_2021 = set()

# Période ancienne (2017-2016)
if 2017 in df_elec_clean['annee'].values:
    communes_periode_2017 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                       2017, 2017, 2016, 2017)
else:
    communes_periode_2017 = set()

# === Trouver les communes ayant les DEUX périodes complètes ===
print("\n🎯 Identification des communes avec les DEUX périodes complètes...")

# Communes ayant 2022 ET 2017
communes_2022_2017 = communes_periode_2022.intersection(communes_periode_2017)
print(f"   - Communes avec 2022 ET 2017: {len(communes_2022_2017)} communes")

# Communes ayant 2021 ET 2017
communes_2021_2017 = communes_periode_2021.intersection(communes_periode_2017)
print(f"   - Communes avec 2021 ET 2017: {len(communes_2021_2017)} communes")

# Toutes les communes valides (union des deux groupes)
communes_valides = communes_2022_2017.union(communes_2021_2017)
print(f"   - 🏆 Total communes valides (avec 2 périodes): {len(communes_valides)} communes")

if len(communes_valides) == 0:
    print("❌ ERREUR: Aucune commune n'a des données complètes pour les deux périodes!")
    print("🔍 Tentative avec une seule période...")
    
    # Si aucune commune n'a deux périodes, prendre la meilleure période disponible
    if len(communes_periode_2022) > 0:
        communes_valides = communes_periode_2022
        print(f"   - Utilisation période 2022: {len(communes_valides)} communes")
    elif len(communes_periode_2021) > 0:
        communes_valides = communes_periode_2021
        print(f"   - Utilisation période 2021: {len(communes_valides)} communes")
    elif len(communes_periode_2017) > 0:
        communes_valides = communes_periode_2017
        print(f"   - Utilisation période 2017: {len(communes_valides)} communes")
    else:
        print("❌ ERREUR CRITIQUE: Aucune période complète trouvée!")
        exit()

# === Fonction pour créer les jointures complètes ===
def creer_jointure_complete(df_elec, df_pop, df_crime, df_emploi, communes_autorisees, 
                           annee_elec, annee_pop, annee_crime, annee_emploi, nom_periode):
    """Crée une jointure complète pour une période donnée"""
    
    print(f"\n🔗 Création jointure {nom_periode}...")
    
    # Filtrer les données pour la période et les communes autorisées
    elec_filtre = df_elec[(df_elec['annee'] == annee_elec) & (df_elec['commune_id'].isin(communes_autorisees))]
    pop_filtre = df_pop[(df_pop['annee'] == annee_pop) & (df_pop['commune_id'].isin(communes_autorisees))]
    crime_filtre = df_crime[(df_crime['annee'] == annee_crime) & (df_crime['commune_id'].isin(communes_autorisees))]
    emploi_filtre = df_emploi[(df_emploi['annee'] == annee_emploi) & (df_emploi['commune_id'].isin(communes_autorisees))]
    
    # Jointures successives (inner join pour garantir la complétude)
    jointure = elec_filtre.merge(pop_filtre.drop(columns='annee'), on='commune_id', how='inner')
    jointure = jointure.merge(crime_filtre.drop(columns='annee'), on='commune_id', how='inner')
    jointure = jointure.merge(emploi_filtre.drop(columns='annee'), on='commune_id', how='inner')
    
    print(f"   - Communes dans jointure {nom_periode}: {len(jointure)} lignes")
    
    # Vérification finale: aucune valeur manquante
    if jointure.isnull().any().any():
        print(f"   - ⚠️ ATTENTION: Valeurs manquantes détectées dans {nom_periode}")
        jointure = jointure.dropna()
        print(f"   - Après suppression des NA: {len(jointure)} lignes")
    
    return jointure

# === Création des jointures pour chaque période ===
print("\n🔨 Création des jointures finales...")

jointures = []

# Jointure 2022 (pour les communes ayant 2022 ET 2017)
if len(communes_2022_2017) > 0:
    join_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                       communes_2022_2017, 2022, 2021, 2021, 2022, "2022")
    jointures.append(join_2022)
    
    join_2017_pour_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                 communes_2022_2017, 2017, 2017, 2016, 2017, "2017 (pour communes 2022)")
    jointures.append(join_2017_pour_2022)

# Jointure 2021 (pour les communes ayant 2021 ET 2017, mais pas dans le groupe 2022)
communes_2021_uniquement = communes_2021_2017 - communes_2022_2017
if len(communes_2021_uniquement) > 0:
    join_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                       communes_2021_uniquement, 2021, 2021, 2021, 2022, "2021")
    jointures.append(join_2021)
    
    join_2017_pour_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                 communes_2021_uniquement, 2017, 2017, 2016, 2017, "2017 (pour communes 2021)")
    jointures.append(join_2017_pour_2021)

# === Fonction pour ordonner les colonnes avec les nouveaux noms ===
def ordonner_colonnes(df):
    """Ordonne les colonnes dans l'ordre souhaité avec les noms professionnels"""
    ordre_colonnes = [
        # Identifiants
        'commune_id',
        'annee',
        # Variables électorales
        'nb_inscrits',
        'nb_abstentions',
        'orientation_politique',
        # Variables démographiques
        'Population_Totale',
        'pct_population_jeune',
        'pct_population_senior',
        'pct_population_sans_activite',
        'pct_population_etrangere',
        # Variables économiques
        'nb_population_active',
        'taux_chomage_pct',
        # Variables sécuritaires
        'nb_crimes'
        # 'taille_commune_categorie',
        # 'nb_emplois_total',
    ]
    
    # Vérifier que toutes les colonnes sont présentes
    colonnes_manquantes = set(ordre_colonnes) - set(df.columns)
    colonnes_excedentaires = set(df.columns) - set(ordre_colonnes)
    
    if colonnes_manquantes:
        print(f"   - ⚠️ Colonnes manquantes: {colonnes_manquantes}")
    if colonnes_excedentaires:
        print(f"   - ⚠️ Colonnes excédentaires: {colonnes_excedentaires}")
        # Ajouter les colonnes excédentaires à la fin
        ordre_colonnes.extend(list(colonnes_excedentaires))
    
    # Réordonner les colonnes (garder seulement celles qui existent)
    colonnes_finales = [col for col in ordre_colonnes if col in df.columns]
    return df[colonnes_finales]

# === Fusion finale ===
if jointures:
    df_joint_final = pd.concat(jointures, ignore_index=True)
    
    # CORRECTION: S'assurer que orientation_politique reste en int après la fusion
    print("\n🔧 Conversion finale orientation_politique en int...")
    df_joint_final['orientation_politique'] = df_joint_final['orientation_politique'].astype('Int64')
    
    # Ordonner les colonnes
    print("\n🔄 Ordonnancement des colonnes...")
    df_joint_final = ordonner_colonnes(df_joint_final)
    
    # Tri par commune et année
    df_joint_final.sort_values(by=['commune_id', 'annee'], ascending=[True, False], inplace=True)
    
    print(f"\n✅ Jointure finale créée:")
    print(f"   - Total lignes: {len(df_joint_final)}")
    print(f"   - Communes uniques: {df_joint_final['commune_id'].nunique()}")
    print(f"   - Années: {sorted(df_joint_final['annee'].unique())}")
    print(f"   - Type orientation_politique: {df_joint_final['orientation_politique'].dtype}")
    
    # Vérification finale: vérifier combien d'enregistrements par commune
    communes_compte = df_joint_final['commune_id'].value_counts()
    communes_avec_2_lignes = (communes_compte == 2).sum()
    communes_avec_1_ligne = (communes_compte == 1).sum()
    communes_problematiques = communes_compte[communes_compte > 2]
    
    print(f"\n🔍 Vérification finale:")
    print(f"   - Communes avec exactement 2 enregistrements: {communes_avec_2_lignes}")
    print(f"   - Communes avec 1 enregistrement: {communes_avec_1_ligne}")
    if len(communes_problematiques) > 0:
        print(f"   - ⚠️ Communes avec plus de 2 enregistrements: {len(communes_problematiques)}")
        print(f"   - Détail: {communes_problematiques.to_dict()}")
        
        # Supprimer les communes problématiques
        communes_valides_finales = communes_compte[communes_compte <= 2].index
        df_joint_final = df_joint_final[df_joint_final['commune_id'].isin(communes_valides_finales)]
        print(f"   - Après correction: {len(df_joint_final)} lignes, {df_joint_final['commune_id'].nunique()} communes")
    
    # Vérification finale des valeurs manquantes
    valeurs_manquantes = df_joint_final.isnull().sum().sum()
    if valeurs_manquantes > 0:
        print(f"   - ⚠️ ATTENTION: {valeurs_manquantes} valeurs manquantes détectées!")
        print("   - Colonnes avec valeurs manquantes:")
        for col in df_joint_final.columns:
            na_count = df_joint_final[col].isnull().sum()
            if na_count > 0:
                print(f"     * {col}: {na_count} valeurs manquantes")
    else:
        print("   - ✅ Aucune valeur manquante détectée!")
    
    # === Sauvegarde ===
    print(f"\n💾 Sauvegarde des fichiers...")
    fichier_excel_final = os.path.join(dossier_sortie, "Dataset_Finale_MSPR_2025_1.xlsx")
    fichier_csv_final = os.path.join(dossier_sortie, "Dataset_Finale_MSPR_2025_1.csv")
    
    df_joint_final.to_excel(fichier_excel_final, index=False)
    df_joint_final.to_csv(fichier_csv_final, index=False, sep=';')
    
    print(f"\n🎉 SUCCÈS! Fichiers sauvegardés:")
    print(f"   📁 Excel: {fichier_excel_final}")
    print(f"   📁 CSV: {fichier_csv_final}")
    print(f"\n📊 Résumé final:")
    print(f"   - {len(df_joint_final)} enregistrements")
    print(f"   - {df_joint_final['commune_id'].nunique()} communes")
    print(f"   - Colonnes finales: {list(df_joint_final.columns)}")
    
    # Afficher un échantillon des données
    print(f"\n📋 Échantillon des données (5 premières lignes):")
    print(df_joint_final.head().to_string())
    
    # === DICTIONNAIRE DE CORRESPONDANCE DES COLONNES ===
    print(f"\n📖 DICTIONNAIRE DE CORRESPONDANCE DES COLONNES:")
    print("="*60)
    correspondance_colonnes = {
        "AVANT": "APRÈS (PROFESSIONNEL)",
        "ID Commune": "commune_id",
        "Année": "annee", 
        "Inscrits": "nb_inscrits_electoral",
        "Abstentions": "nb_abstentions",
        "Score Orientation (0 à 4)": "score_orientation_politique",
        "Pct_Jeunes": "pct_population_jeune",
        "Pct_Seniors": "pct_population_senior", 
        "Pct_Sans_Activite": "pct_population_sans_activite",
        "Pct_Etrangere": "pct_population_etrangere",
        # "Taille_Commune": "taille_commune_categorie",
        "Population Active": "nb_population_active",
        # "Emplois": "nb_emplois_total",
        "% Chômage": "taux_chomage_pct",
        "nb_crimes": "nb_crimes_total"
    }
    
    for avant, apres in correspondance_colonnes.items():
        print(f"   {avant:<30} → {apres}")
    
else:
    print("❌ ERREUR: Aucune jointure n'a pu être créée!")