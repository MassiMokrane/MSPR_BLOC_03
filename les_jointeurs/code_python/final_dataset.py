# import pandas as pd
# import os
# import numpy as np

# # === Chemins ===
# fichier_elections = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Nettoyer_Elections.csv"
# fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
# fichier_criminaliter = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Criminalite_2021_2016.csv"
# fichier_emploi = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Emploi_2022_2017.csv"
# dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/"
# os.makedirs(dossier_sortie, exist_ok=True)

# print("🔄 Lecture des données...")
# # === Lecture des données ===
# df_elec_full = pd.read_csv(fichier_elections, sep=';')
# df_pop_full = pd.read_csv(fichier_population, sep=',')
# df_crime_full = pd.read_csv(fichier_criminaliter, sep=';')
# df_emploi_full = pd.read_csv(fichier_emploi, sep=';')

# # === Sélection des colonnes spécifiées ===
# print("📋 Sélection des colonnes...")

# # Elections: ID Commune;Année;Inscrits;Abstentions;Orientation Politique;Score Orientation (0 à 4)
# colonnes_elec = ['ID Commune', 'Année', 'Inscrits', 'Abstentions', 'Orientation Politique', 'Score Orientation (0 à 4)']
# df_elec = df_elec_full[colonnes_elec].copy()

# # Population: ID Commune + année + colonnes sélectionnées
# colonnes_pop = ['ID Commune', 'année', 'pop totale', 'pop de 0 à 19ans', 'pop de 20 à 64ans', 'pop 65ans ou plus', 'pop Retraités', 'pop en ménages', 'pop hors ménages']
# df_pop = df_pop_full[colonnes_pop].copy()

# # Criminalité: ID Commune + Année + nb_crimes
# colonnes_crime = ['ID Commune', 'Année', 'nb_crimes']
# df_crime = df_crime_full[colonnes_crime].copy()

# # Emploi: ID Commune + Année + colonnes sélectionnées
# colonnes_emploi = ['ID Commune', 'Année', 'Population Active', 'Chômeurs', 'Emplois', '% Chômage']
# df_emploi = df_emploi_full[colonnes_emploi].copy()

# print(f"📊 Données chargées:")
# print(f"   - Elections: {len(df_elec)} lignes, {len(df_elec.columns)} colonnes")
# print(f"   - Population: {len(df_pop)} lignes, {len(df_pop.columns)} colonnes")
# print(f"   - Criminalité: {len(df_crime)} lignes, {len(df_crime.columns)} colonnes")
# print(f"   - Emploi: {len(df_emploi)} lignes, {len(df_emploi.columns)} colonnes")

# # === Nettoyage des IDs ===
# for df in [df_elec, df_pop, df_crime, df_emploi]:
#     df['ID Commune'] = df['ID Commune'].astype(str).str.zfill(5)

# # === Filtrer jusqu'à ID Commune 95690 inclus ===
# df_elec = df_elec[df_elec['ID Commune'] <= '95690']
# df_pop = df_pop[df_pop['ID Commune'] <= '95690']
# df_crime = df_crime[df_crime['ID Commune'] <= '95690']
# df_emploi = df_emploi[df_emploi['ID Commune'] <= '95690']

# # === Convertir les années ===
# df_elec['Année'] = df_elec['Année'].astype(int)
# df_pop['année'] = df_pop['année'].astype(int)
# df_crime['Année'] = df_crime['Année'].astype(int)
# df_emploi['Année'] = df_emploi['Année'].astype(int)

# print("\n🔍 Vérification de la complétude des données par commune...")

# # === Fonction pour vérifier la complétude d'un DataFrame ===
# def verifier_completude(df, nom_dataset):
#     """Vérifie qu'un DataFrame n'a pas de valeurs manquantes"""
#     lignes_avec_na = df.isnull().any(axis=1).sum()
#     print(f"   - {nom_dataset}: {lignes_avec_na} lignes avec valeurs manquantes")
#     return df.dropna()

# # === Nettoyage des valeurs manquantes ===
# print("\n🧹 Suppression des lignes avec valeurs manquantes...")
# df_elec_clean = verifier_completude(df_elec, "Elections")
# df_pop_clean = verifier_completude(df_pop, "Population")
# df_crime_clean = verifier_completude(df_crime, "Criminalité")
# df_emploi_clean = verifier_completude(df_emploi, "Emploi")


# print(f"\n📋 Données après nettoyage:")
# print(f"   - Elections: {len(df_elec_clean)} lignes")
# print(f"   - Population: {len(df_pop_clean)} lignes")
# print(f"   - Criminalité: {len(df_crime_clean)} lignes")
# print(f"   - Emploi: {len(df_emploi_clean)} lignes")

# # === Identifier les communes avec données complètes pour chaque période ===

# def trouver_communes_completes(df_elec, df_pop, df_crime, df_emploi, annee_elec, annee_pop, annee_crime, annee_emploi):
#     """Trouve les communes ayant des données complètes pour toutes les sources pour une période donnée"""
    
#     communes_elec = set(df_elec[df_elec['Année'] == annee_elec]['ID Commune'])
#     communes_pop = set(df_pop[df_pop['année'] == annee_pop]['ID Commune'])
#     communes_crime = set(df_crime[df_crime['Année'] == annee_crime]['ID Commune'])
#     communes_emploi = set(df_emploi[df_emploi['Année'] == annee_emploi]['ID Commune'])
    
#     # Intersection de toutes les communes (présentes dans tous les datasets)
#     communes_completes = communes_elec.intersection(communes_pop).intersection(communes_crime).intersection(communes_emploi)
    
#     print(f"\n📈 Période {annee_elec} (pop:{annee_pop}, crime:{annee_crime}, emploi:{annee_emploi}):")
#     print(f"   - Elections: {len(communes_elec)} communes")
#     print(f"   - Population: {len(communes_pop)} communes")
#     print(f"   - Criminalité: {len(communes_crime)} communes")
#     print(f"   - Emploi: {len(communes_emploi)} communes")
#     print(f"   - ✅ Communes complètes: {len(communes_completes)} communes")
    
#     return communes_completes

# # === Trouver les communes complètes pour chaque période ===
# print("\n🔎 Identification des communes avec données complètes...")

# # Période récente (2021-2022)
# communes_periode_2022 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2022, 2021, 2021, 2022)

# communes_periode_2021 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2021, 2021, 2021, 2022)

# # Période ancienne (2016-2017)
# communes_periode_2017 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2017, 2017, 2016, 2017)

# # === Trouver les communes ayant les DEUX périodes complètes ===
# print("\n🎯 Identification des communes avec les DEUX périodes complètes...")

# # Communes ayant 2022 ET 2017
# communes_2022_2017 = communes_periode_2022.intersection(communes_periode_2017)
# print(f"   - Communes avec 2022 ET 2017: {len(communes_2022_2017)} communes")

# # Communes ayant 2021 ET 2017
# communes_2021_2017 = communes_periode_2021.intersection(communes_periode_2017)
# print(f"   - Communes avec 2021 ET 2017: {len(communes_2021_2017)} communes")

# # Toutes les communes valides (union des deux groupes)
# communes_valides = communes_2022_2017.union(communes_2021_2017)
# print(f"   - 🏆 Total communes valides (avec 2 périodes): {len(communes_valides)} communes")

# if len(communes_valides) == 0:
#     print("❌ ERREUR: Aucune commune n'a des données complètes pour les deux périodes!")
#     exit()

# # === Fonction pour créer les jointures complètes ===
# def creer_jointure_complete(df_elec, df_pop, df_crime, df_emploi, communes_autorisees, 
#                            annee_elec, annee_pop, annee_crime, annee_emploi, nom_periode):
#     """Crée une jointure complète pour une période donnée"""
    
#     print(f"\n🔗 Création jointure {nom_periode}...")
    
#     # Filtrer les données pour la période et les communes autorisées
#     elec_filtre = df_elec[(df_elec['Année'] == annee_elec) & (df_elec['ID Commune'].isin(communes_autorisees))]
#     pop_filtre = df_pop[(df_pop['année'] == annee_pop) & (df_pop['ID Commune'].isin(communes_autorisees))]
#     crime_filtre = df_crime[(df_crime['Année'] == annee_crime) & (df_crime['ID Commune'].isin(communes_autorisees))]
#     emploi_filtre = df_emploi[(df_emploi['Année'] == annee_emploi) & (df_emploi['ID Commune'].isin(communes_autorisees))]
    
#     # Jointures successives (inner join pour garantir la complétude)
#     jointure = elec_filtre.merge(pop_filtre, on='ID Commune', how='inner')
#     jointure = jointure.merge(crime_filtre.drop(columns='Année'), on='ID Commune', how='inner')
#     jointure = jointure.merge(emploi_filtre.drop(columns='Année'), on='ID Commune', how='inner')
    
#     print(f"   - Communes dans jointure {nom_periode}: {len(jointure)} lignes")
    
#     # Vérification finale: aucune valeur manquante
#     if jointure.isnull().any().any():
#         print(f"   - ⚠️ ATTENTION: Valeurs manquantes détectées dans {nom_periode}")
#         jointure = jointure.dropna()
#         print(f"   - Après suppression des NA: {len(jointure)} lignes")
    
#     return jointure

# # === Création des jointures pour chaque période ===
# print("\n🔨 Création des jointures finales...")

# jointures = []

# # Jointure 2022 (pour les communes ayant 2022 ET 2017)
# if len(communes_2022_2017) > 0:
#     join_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                        communes_2022_2017, 2022, 2021, 2021, 2022, "2022")
#     jointures.append(join_2022)
    
#     join_2017_pour_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                  communes_2022_2017, 2017, 2017, 2016, 2017, "2017 (pour communes 2022)")
#     jointures.append(join_2017_pour_2022)

# # Jointure 2021 (pour les communes ayant 2021 ET 2017, mais pas dans le groupe 2022)
# communes_2021_uniquement = communes_2021_2017 - communes_2022_2017
# if len(communes_2021_uniquement) > 0:
#     join_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                        communes_2021_uniquement, 2021, 2021, 2021, 2022, "2021")
#     jointures.append(join_2021)
    
#     join_2017_pour_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                  communes_2021_uniquement, 2017, 2017, 2016, 2017, "2017 (pour communes 2021)")
#     jointures.append(join_2017_pour_2021)

# # === Fusion finale ===
# if jointures:
#     df_joint_final = pd.concat(jointures, ignore_index=True)
    
#     # Tri par commune et année
#     df_joint_final.sort_values(by=['ID Commune', 'Année'], ascending=[True, False], inplace=True)
    
#     print(f"\n✅ Jointure finale créée:")
#     print(f"   - Total lignes: {len(df_joint_final)}")
#     print(f"   - Communes uniques: {df_joint_final['ID Commune'].nunique()}")
#     print(f"   - Années: {sorted(df_joint_final['Année'].unique())}")
    
#     # Vérification finale: chaque commune a bien 2 enregistrements
#     communes_compte = df_joint_final['ID Commune'].value_counts()
#     communes_avec_2_lignes = (communes_compte == 2).sum()
#     communes_problematiques = communes_compte[communes_compte != 2]
    
#     print(f"\n🔍 Vérification finale:")
#     print(f"   - Communes avec exactement 2 enregistrements: {communes_avec_2_lignes}")
#     if len(communes_problematiques) > 0:
#         print(f"   - ⚠️ Communes problématiques: {len(communes_problematiques)}")
#         print(f"   - Détail: {communes_problematiques.to_dict()}")
        
#         # Supprimer les communes problématiques
#         communes_a_garder = communes_compte[communes_compte == 2].index
#         df_joint_final = df_joint_final[df_joint_final['ID Commune'].isin(communes_a_garder)]
#         print(f"   - Après correction: {len(df_joint_final)} lignes, {df_joint_final['ID Commune'].nunique()} communes")
    
#     # Vérification finale des valeurs manquantes
#     valeurs_manquantes = df_joint_final.isnull().sum().sum()
#     if valeurs_manquantes > 0:
#         print(f"   - ⚠️ ATTENTION: {valeurs_manquantes} valeurs manquantes détectées!")
#         print("   - Colonnes avec valeurs manquantes:")
#         for col in df_joint_final.columns:
#             na_count = df_joint_final[col].isnull().sum()
#             if na_count > 0:
#                 print(f"     * {col}: {na_count} valeurs manquantes")
#     else:
#         print("   - ✅ Aucune valeur manquante détectée!")
    
#     # === Sauvegarde ===
#     print(f"\n💾 Sauvegarde des fichiers...")
#     fichier_excel_final = os.path.join(dossier_sortie, "Dataset_Finale_all.xlsx")
#     fichier_csv_final = os.path.join(dossier_sortie, "Dataset_Finale_all.csv")
    
#     df_joint_final.to_excel(fichier_excel_final, index=False)
#     df_joint_final.to_csv(fichier_csv_final, index=False)
    
#     print(f"\n🎉 SUCCÈS! Fichiers sauvegardés:")
#     print(f"   📁 Excel: {fichier_excel_final}")
#     print(f"   📁 CSV: {fichier_csv_final}")
#     print(f"\n📊 Résumé final:")
#     print(f"   - {len(df_joint_final)} enregistrements")
#     print(f"   - {df_joint_final['ID Commune'].nunique()} communes")
#     print(f"   - Chaque commune a exactement 2 enregistrements (2 périodes)")
#     print(f"   - Colonnes finales: {list(df_joint_final.columns)}")
    
# else:
#     print("❌ ERREUR: Aucune jointure n'a pu être créée!")


# #best for the moments ******************************************************************************************************************************************************************************************************************************
# import pandas as pd
# import os
# import numpy as np

# # === Chemins ===
# fichier_elections = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Nettoyer_Elections.csv"
# fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
# fichier_criminaliter = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Criminalite_2021_2016.csv"
# fichier_emploi = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Emploi_2022_2017.csv"
# dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/"
# os.makedirs(dossier_sortie, exist_ok=True)

# print("🔄 Lecture des données...")
# # === Lecture des données ===
# df_elec_full = pd.read_csv(fichier_elections, sep=';')
# df_pop_full = pd.read_csv(fichier_population, sep=',')
# df_crime_full = pd.read_csv(fichier_criminaliter, sep=';')
# df_emploi_full = pd.read_csv(fichier_emploi, sep=';')

# # === Sélection des colonnes spécifiées ===
# print("📋 Sélection des colonnes...")

# # Elections: ID Commune;Année;Inscrits;Abstentions;Orientation Politique;Score Orientation (0 à 4)
# colonnes_elec = ['ID Commune', 'Année', 'Inscrits', 'Abstentions', 'Orientation Politique', 'Score Orientation (0 à 4)']
# df_elec = df_elec_full[colonnes_elec].copy()

# # Population: ID Commune + année + colonnes sélectionnées (année sera supprimée après jointure)
# colonnes_pop = ['ID Commune', 'année', 'pop totale', 'pop de 0 à 19ans', 'pop de 20 à 64ans', 'pop 65ans ou plus', 'pop Retraités', 'pop en ménages', 'pop hors ménages']
# df_pop = df_pop_full[colonnes_pop].copy()

# # Criminalité: ID Commune + Année + nb_crimes
# colonnes_crime = ['ID Commune', 'Année', 'nb_crimes']
# df_crime = df_crime_full[colonnes_crime].copy()

# # Emploi: ID Commune + Année + colonnes sélectionnées
# colonnes_emploi = ['ID Commune', 'Année', 'Population Active', 'Chômeurs', 'Emplois', '% Chômage']
# df_emploi = df_emploi_full[colonnes_emploi].copy()

# print(f"📊 Données chargées:")
# print(f"   - Elections: {len(df_elec)} lignes, {len(df_elec.columns)} colonnes")
# print(f"   - Population: {len(df_pop)} lignes, {len(df_pop.columns)} colonnes")
# print(f"   - Criminalité: {len(df_crime)} lignes, {len(df_crime.columns)} colonnes")
# print(f"   - Emploi: {len(df_emploi)} lignes, {len(df_emploi.columns)} colonnes")

# # === Nettoyage des IDs ===
# for df in [df_elec, df_pop, df_crime, df_emploi]:
#     df['ID Commune'] = df['ID Commune'].astype(str).str.zfill(5)

# # === Filtrer jusqu'à ID Commune 95690 inclus ===
# df_elec = df_elec[df_elec['ID Commune'] <= '95690']
# df_pop = df_pop[df_pop['ID Commune'] <= '95690']
# df_crime = df_crime[df_crime['ID Commune'] <= '95690']
# df_emploi = df_emploi[df_emploi['ID Commune'] <= '95690']

# # === Convertir les années ===
# df_elec['Année'] = df_elec['Année'].astype(int)
# df_pop['année'] = df_pop['année'].astype(int)
# df_crime['Année'] = df_crime['Année'].astype(int)
# df_emploi['Année'] = df_emploi['Année'].astype(int)

# print("\n🔍 Vérification de la complétude des données par commune...")

# # === Fonction pour vérifier la complétude d'un DataFrame ===
# def verifier_completude(df, nom_dataset):
#     """Vérifie qu'un DataFrame n'a pas de valeurs manquantes"""
#     lignes_avec_na = df.isnull().any(axis=1).sum()
#     print(f"   - {nom_dataset}: {lignes_avec_na} lignes avec valeurs manquantes")
#     return df.dropna()

# # === Nettoyage des valeurs manquantes ===
# print("\n🧹 Suppression des lignes avec valeurs manquantes...")
# df_elec_clean = verifier_completude(df_elec, "Elections")
# df_pop_clean = verifier_completude(df_pop, "Population")
# df_crime_clean = verifier_completude(df_crime, "Criminalité")
# df_emploi_clean = verifier_completude(df_emploi, "Emploi")


# print(f"\n📋 Données après nettoyage:")
# print(f"   - Elections: {len(df_elec_clean)} lignes")
# print(f"   - Population: {len(df_pop_clean)} lignes")
# print(f"   - Criminalité: {len(df_crime_clean)} lignes")
# print(f"   - Emploi: {len(df_emploi_clean)} lignes")

# # === Identifier les communes avec données complètes pour chaque période ===

# def trouver_communes_completes(df_elec, df_pop, df_crime, df_emploi, annee_elec, annee_pop, annee_crime, annee_emploi):
#     """Trouve les communes ayant des données complètes pour toutes les sources pour une période donnée"""
    
#     communes_elec = set(df_elec[df_elec['Année'] == annee_elec]['ID Commune'])
#     communes_pop = set(df_pop[df_pop['année'] == annee_pop]['ID Commune'])
#     communes_crime = set(df_crime[df_crime['Année'] == annee_crime]['ID Commune'])
#     communes_emploi = set(df_emploi[df_emploi['Année'] == annee_emploi]['ID Commune'])
    
#     # Intersection de toutes les communes (présentes dans tous les datasets)
#     communes_completes = communes_elec.intersection(communes_pop).intersection(communes_crime).intersection(communes_emploi)
    
#     print(f"\n📈 Période {annee_elec} (pop:{annee_pop}, crime:{annee_crime}, emploi:{annee_emploi}):")
#     print(f"   - Elections: {len(communes_elec)} communes")
#     print(f"   - Population: {len(communes_pop)} communes")
#     print(f"   - Criminalité: {len(communes_crime)} communes")
#     print(f"   - Emploi: {len(communes_emploi)} communes")
#     print(f"   - ✅ Communes complètes: {len(communes_completes)} communes")
    
#     return communes_completes

# # === Trouver les communes complètes pour chaque période ===
# print("\n🔎 Identification des communes avec données complètes...")

# # Période récente (2021-2022)
# communes_periode_2022 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2022, 2021, 2021, 2022)

# communes_periode_2021 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2021, 2021, 2021, 2022)

# # Période ancienne (2016-2017)
# communes_periode_2017 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                    2017, 2017, 2016, 2017)

# # === Trouver les communes ayant les DEUX périodes complètes ===
# print("\n🎯 Identification des communes avec les DEUX périodes complètes...")

# # Communes ayant 2022 ET 2017
# communes_2022_2017 = communes_periode_2022.intersection(communes_periode_2017)
# print(f"   - Communes avec 2022 ET 2017: {len(communes_2022_2017)} communes")

# # Communes ayant 2021 ET 2017
# communes_2021_2017 = communes_periode_2021.intersection(communes_periode_2017)
# print(f"   - Communes avec 2021 ET 2017: {len(communes_2021_2017)} communes")

# # Toutes les communes valides (union des deux groupes)
# communes_valides = communes_2022_2017.union(communes_2021_2017)
# print(f"   - 🏆 Total communes valides (avec 2 périodes): {len(communes_valides)} communes")

# if len(communes_valides) == 0:
#     print("❌ ERREUR: Aucune commune n'a des données complètes pour les deux périodes!")
#     exit()

# # === Fonction pour créer les jointures complètes ===
# def creer_jointure_complete(df_elec, df_pop, df_crime, df_emploi, communes_autorisees, 
#                            annee_elec, annee_pop, annee_crime, annee_emploi, nom_periode):
#     """Crée une jointure complète pour une période donnée"""
    
#     print(f"\n🔗 Création jointure {nom_periode}...")
    
#     # Filtrer les données pour la période et les communes autorisées
#     elec_filtre = df_elec[(df_elec['Année'] == annee_elec) & (df_elec['ID Commune'].isin(communes_autorisees))]
#     pop_filtre = df_pop[(df_pop['année'] == annee_pop) & (df_pop['ID Commune'].isin(communes_autorisees))]
#     crime_filtre = df_crime[(df_crime['Année'] == annee_crime) & (df_crime['ID Commune'].isin(communes_autorisees))]
#     emploi_filtre = df_emploi[(df_emploi['Année'] == annee_emploi) & (df_emploi['ID Commune'].isin(communes_autorisees))]
    
#     # Jointures successives (inner join pour garantir la complétude)
#     jointure = elec_filtre.merge(pop_filtre.drop(columns='année'), on='ID Commune', how='inner')
#     jointure = jointure.merge(crime_filtre.drop(columns='Année'), on='ID Commune', how='inner')
#     jointure = jointure.merge(emploi_filtre.drop(columns='Année'), on='ID Commune', how='inner')
    
#     print(f"   - Communes dans jointure {nom_periode}: {len(jointure)} lignes")
    
#     # Vérification finale: aucune valeur manquante
#     if jointure.isnull().any().any():
#         print(f"   - ⚠️ ATTENTION: Valeurs manquantes détectées dans {nom_periode}")
#         jointure = jointure.dropna()
#         print(f"   - Après suppression des NA: {len(jointure)} lignes")
    
#     return jointure

# # === Création des jointures pour chaque période ===
# print("\n🔨 Création des jointures finales...")

# jointures = []

# # Jointure 2022 (pour les communes ayant 2022 ET 2017)
# if len(communes_2022_2017) > 0:
#     join_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                        communes_2022_2017, 2022, 2021, 2021, 2022, "2022")
#     jointures.append(join_2022)
    
#     join_2017_pour_2022 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                  communes_2022_2017, 2017, 2017, 2016, 2017, "2017 (pour communes 2022)")
#     jointures.append(join_2017_pour_2022)

# # Jointure 2021 (pour les communes ayant 2021 ET 2017, mais pas dans le groupe 2022)
# communes_2021_uniquement = communes_2021_2017 - communes_2022_2017
# if len(communes_2021_uniquement) > 0:
#     join_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                        communes_2021_uniquement, 2021, 2021, 2021, 2022, "2021")
#     jointures.append(join_2021)
    
#     join_2017_pour_2021 = creer_jointure_complete(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
#                                                  communes_2021_uniquement, 2017, 2017, 2016, 2017, "2017 (pour communes 2021)")
#     jointures.append(join_2017_pour_2021)

# # === Fusion finale ===
# if jointures:
#     df_joint_final = pd.concat(jointures, ignore_index=True)
    
#     # Tri par commune et année
#     df_joint_final.sort_values(by=['ID Commune', 'Année'], ascending=[True, False], inplace=True)
    
#     print(f"\n✅ Jointure finale créée:")
#     print(f"   - Total lignes: {len(df_joint_final)}")
#     print(f"   - Communes uniques: {df_joint_final['ID Commune'].nunique()}")
#     print(f"   - Années: {sorted(df_joint_final['Année'].unique())}")
    
#     # Vérification finale: chaque commune a bien 2 enregistrements
#     communes_compte = df_joint_final['ID Commune'].value_counts()
#     communes_avec_2_lignes = (communes_compte == 2).sum()
#     communes_problematiques = communes_compte[communes_compte != 2]
    
#     print(f"\n🔍 Vérification finale:")
#     print(f"   - Communes avec exactement 2 enregistrements: {communes_avec_2_lignes}")
#     if len(communes_problematiques) > 0:
#         print(f"   - ⚠️ Communes problématiques: {len(communes_problematiques)}")
#         print(f"   - Détail: {communes_problematiques.to_dict()}")
        
#         # Supprimer les communes problématiques
#         communes_a_garder = communes_compte[communes_compte == 2].index
#         df_joint_final = df_joint_final[df_joint_final['ID Commune'].isin(communes_a_garder)]
#         print(f"   - Après correction: {len(df_joint_final)} lignes, {df_joint_final['ID Commune'].nunique()} communes")
    
#     # Vérification finale des valeurs manquantes
#     valeurs_manquantes = df_joint_final.isnull().sum().sum()
#     if valeurs_manquantes > 0:
#         print(f"   - ⚠️ ATTENTION: {valeurs_manquantes} valeurs manquantes détectées!")
#         print("   - Colonnes avec valeurs manquantes:")
#         for col in df_joint_final.columns:
#             na_count = df_joint_final[col].isnull().sum()
#             if na_count > 0:
#                 print(f"     * {col}: {na_count} valeurs manquantes")
#     else:
#         print("   - ✅ Aucune valeur manquante détectée!")
    
#     # === Sauvegarde ===
#     print(f"\n💾 Sauvegarde des fichiers...")
#     fichier_excel_final = os.path.join(dossier_sortie, "Dataset_Finale.xlsx")
#     fichier_csv_final = os.path.join(dossier_sortie, "Dataset_Finale.csv")
    
#     df_joint_final.to_excel(fichier_excel_final, index=False)
#     df_joint_final.to_csv(fichier_csv_final, index=False)
    
#     print(f"\n🎉 SUCCÈS! Fichiers sauvegardés:")
#     print(f"   📁 Excel: {fichier_excel_final}")
#     print(f"   📁 CSV: {fichier_csv_final}")
#     print(f"\n📊 Résumé final:")
#     print(f"   - {len(df_joint_final)} enregistrements")
#     print(f"   - {df_joint_final['ID Commune'].nunique()} communes")
#     print(f"   - Chaque commune a exactement 2 enregistrements (2 périodes)")
#     print(f"   - Colonnes finales: {list(df_joint_final.columns)}")
    
# else:
#     print("❌ ERREUR: Aucune jointure n'a pu être créée!")












# same with pecedent with colonne ordering *******************************************************************************************************************************************************************************************************
import pandas as pd
import os
import numpy as np

# === Chemins ===
fichier_elections = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Nettoyer_Elections.csv"
fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
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

# === Sélection des colonnes spécifiées ===
print("📋 Sélection des colonnes...")

# Elections: ID Commune;Année;Inscrits;Abstentions;Orientation Politique;Score Orientation (0 à 4)
colonnes_elec = ['ID Commune', 'Année', 'Inscrits', 'Abstentions', 'Orientation Politique'
                #  , 'Score Orientation (0 à 4)'
                 ]
df_elec = df_elec_full[colonnes_elec].copy()

# Population: ID Commune + année + colonnes sélectionnées (année sera supprimée après jointure)
colonnes_pop = ['ID Commune', 'année', 'pop totale', 
                'pop de 0 à 19ans', 
                'pop de 20 à 64ans',
                  'pop 65ans ou plus',
                    'pop Retraités', 
                    'pop en ménages', 
                    'pop hors ménages']
df_pop = df_pop_full[colonnes_pop].copy()

# Criminalité: ID Commune + Année + nb_crimes
colonnes_crime = ['ID Commune', 'Année', 'nb_crimes']
df_crime = df_crime_full[colonnes_crime].copy()

# Emploi: ID Commune + Année + colonnes sélectionnées
colonnes_emploi = ['ID Commune',
                    'Année', 
                    'Population Active', 
                    'Chômeurs',
                    'Emplois',
                    '% Chômage']
df_emploi = df_emploi_full[colonnes_emploi].copy()

print(f"📊 Données chargées:")
print(f"   - Elections: {len(df_elec)} lignes, {len(df_elec.columns)} colonnes")
print(f"   - Population: {len(df_pop)} lignes, {len(df_pop.columns)} colonnes")
print(f"   - Criminalité: {len(df_crime)} lignes, {len(df_crime.columns)} colonnes")
print(f"   - Emploi: {len(df_emploi)} lignes, {len(df_emploi.columns)} colonnes")

# === Nettoyage des IDs ===
for df in [df_elec, df_pop, df_crime, df_emploi]:
    df['ID Commune'] = df['ID Commune'].astype(str).str.zfill(5)

# === Filtrer jusqu'à ID Commune 95690 inclus ===
df_elec = df_elec[df_elec['ID Commune'] <= '95690']
df_pop = df_pop[df_pop['ID Commune'] <= '95690']
df_crime = df_crime[df_crime['ID Commune'] <= '95690']
df_emploi = df_emploi[df_emploi['ID Commune'] <= '95690']

# === Convertir les années ===
df_elec['Année'] = df_elec['Année'].astype(int)
df_pop['année'] = df_pop['année'].astype(int)
df_crime['Année'] = df_crime['Année'].astype(int)
df_emploi['Année'] = df_emploi['Année'].astype(int)

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
    
    communes_elec = set(df_elec[df_elec['Année'] == annee_elec]['ID Commune'])
    communes_pop = set(df_pop[df_pop['année'] == annee_pop]['ID Commune'])
    communes_crime = set(df_crime[df_crime['Année'] == annee_crime]['ID Commune'])
    communes_emploi = set(df_emploi[df_emploi['Année'] == annee_emploi]['ID Commune'])
    
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

# Période récente (2021-2022)
communes_periode_2022 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                   2022, 2021, 2021, 2022)

communes_periode_2021 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                   2021, 2021, 2021, 2022)

# Période ancienne (2016-2017)
communes_periode_2017 = trouver_communes_completes(df_elec_clean, df_pop_clean, df_crime_clean, df_emploi_clean, 
                                                   2017, 2017, 2016, 2017)

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
    exit()

# === Fonction pour créer les jointures complètes ===
def creer_jointure_complete(df_elec, df_pop, df_crime, df_emploi, communes_autorisees, 
                           annee_elec, annee_pop, annee_crime, annee_emploi, nom_periode):
    """Crée une jointure complète pour une période donnée"""
    
    print(f"\n🔗 Création jointure {nom_periode}...")
    
    # Filtrer les données pour la période et les communes autorisées
    elec_filtre = df_elec[(df_elec['Année'] == annee_elec) & (df_elec['ID Commune'].isin(communes_autorisees))]
    pop_filtre = df_pop[(df_pop['année'] == annee_pop) & (df_pop['ID Commune'].isin(communes_autorisees))]
    crime_filtre = df_crime[(df_crime['Année'] == annee_crime) & (df_crime['ID Commune'].isin(communes_autorisees))]
    emploi_filtre = df_emploi[(df_emploi['Année'] == annee_emploi) & (df_emploi['ID Commune'].isin(communes_autorisees))]
    
    # Jointures successives (inner join pour garantir la complétude)
    jointure = elec_filtre.merge(pop_filtre.drop(columns='année'), on='ID Commune', how='inner')
    jointure = jointure.merge(crime_filtre.drop(columns='Année'), on='ID Commune', how='inner')
    jointure = jointure.merge(emploi_filtre.drop(columns='Année'), on='ID Commune', how='inner')
    
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

# === Fonction pour ordonner les colonnes ===
def ordonner_colonnes(df):
    """Ordonne les colonnes dans l'ordre souhaité"""
    ordre_colonnes = [
        'ID Commune',
        'Année',
        # Colonnes Elections
        'Inscrits',
        'Abstentions', 
        'Orientation Politique',
        'Score Orientation (0 à 4)',
        # Colonnes Population
        'pop totale',
        'pop de 0 à 19ans',
        'pop de 20 à 64ans',
        'pop 65ans ou plus',
        'pop Retraités',
        'pop en ménages',
        'pop hors ménages',
        # Colonnes Emploi
        'Population Active',
        'Emplois',
        'Chômeurs',
        '% Chômage',
        # Colonnes Criminalité
        'nb_crimes'
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
    
    # Ordonner les colonnes
    print("\n🔄 Ordonnancement des colonnes...")
    df_joint_final = ordonner_colonnes(df_joint_final)
    
    # Tri par commune et année
    df_joint_final.sort_values(by=['ID Commune', 'Année'], ascending=[True, False], inplace=True)
    
    print(f"\n✅ Jointure finale créée:")
    print(f"   - Total lignes: {len(df_joint_final)}")
    print(f"   - Communes uniques: {df_joint_final['ID Commune'].nunique()}")
    print(f"   - Années: {sorted(df_joint_final['Année'].unique())}")
    
    # Vérification finale: chaque commune a bien 2 enregistrements
    communes_compte = df_joint_final['ID Commune'].value_counts()
    communes_avec_2_lignes = (communes_compte == 2).sum()
    communes_problematiques = communes_compte[communes_compte != 2]
    
    print(f"\n🔍 Vérification finale:")
    print(f"   - Communes avec exactement 2 enregistrements: {communes_avec_2_lignes}")
    if len(communes_problematiques) > 0:
        print(f"   - ⚠️ Communes problématiques: {len(communes_problematiques)}")
        print(f"   - Détail: {communes_problematiques.to_dict()}")
        
        # Supprimer les communes problématiques
        communes_a_garder = communes_compte[communes_compte == 2].index
        df_joint_final = df_joint_final[df_joint_final['ID Commune'].isin(communes_a_garder)]
        print(f"   - Après correction: {len(df_joint_final)} lignes, {df_joint_final['ID Commune'].nunique()} communes")
    
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
    fichier_excel_final = os.path.join(dossier_sortie, "Dataset_Finale_MSPR.xlsx")
    fichier_csv_final = os.path.join(dossier_sortie, "Dataset_Finale_MSPR.csv")
    
    df_joint_final.to_excel(fichier_excel_final, index=False)
    df_joint_final.to_csv(fichier_csv_final, index=False)
    
    print(f"\n🎉 SUCCÈS! Fichiers sauvegardés:")
    print(f"   📁 Excel: {fichier_excel_final}")
    print(f"   📁 CSV: {fichier_csv_final}")
    print(f"\n📊 Résumé final:")
    print(f"   - {len(df_joint_final)} enregistrements")
    print(f"   - {df_joint_final['ID Commune'].nunique()} communes")
    print(f"   - Chaque commune a exactement 2 enregistrements (2 périodes)")
    print(f"   - Colonnes finales: {list(df_joint_final.columns)}")
    
else:
    print("❌ ERREUR: Aucune jointure n'a pu être créée!")