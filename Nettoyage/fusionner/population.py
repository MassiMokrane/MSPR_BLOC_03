# # import os
# # import pandas as pd
# # import numpy as np

# # # === CONFIGURATION DES CHEMINS ===
# # fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
# # dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Population"

# # print("=== NETTOYAGE DATASET POPULATION ===")
# # print("=" * 50)

# # def nettoyer_dataset_population(fichier_entree, dossier_sortie):
# #     """
# #     Nettoie le dataset population en gardant seulement les colonnes essentielles
# #     """
    
# #     # === CHARGEMENT ===
# #     try:
# #         print("📂 Chargement du dataset population...")
# #         df = pd.read_csv(fichier_entree)
# #         print(f"   Dataset original: {df.shape[0]} lignes × {df.shape[1]} colonnes")
# #     except FileNotFoundError:
# #         print(f"❌ Fichier non trouvé : {fichier_entree}")
# #         return None
# #     except Exception as e:
# #         print(f"❌ Erreur de chargement : {e}")
# #         return None
    
# #     # === RENOMMAGE DES COLONNES ===
# #     print("🔧 Renommage des colonnes...")
    
# #     # Mapping des colonnes principales
# #     column_mapping = {
# #         'ID Commune': 'ID_Commune',
# #         'pop totale': 'Population_Totale',
# #         'pop de 0 à 19ans': 'Pop_0_19',
# #         'pop de 20 à 64ans': 'Pop_20_64',
# #         'pop 65ans ou plus': 'Pop_65_Plus',
# #         'pop 15 ans ou plus Agriculteurs': 'Agriculteurs',
# #         'pop 15 ans ou plus Artisans, Comm': 'Artisans_Commercants',
# #         'pop 15 ans ou plus Cadres, Prof': 'Cadres_Prof',
# #         'pop 15 ans ou plus Prof. intermédiaires': 'Prof_Intermediaires',
# #         'pop 15 ans ou plus Employés': 'Employes',
# #         'pop 15 ans ou plus Ouvriers': 'Ouvriers',
# #         'pop Retraités': 'Retraites',
# #         'pop 15 ans ou plus sans activité professionnelle': 'Sans_Activite',
# #         'pop française': 'Pop_Francaise',
# #         'pop étrangère': 'Pop_Etrangere',
# #         'pop immigrée': 'Pop_Immigree',
# #         'pop en ménages': 'Pop_Menages',
# #         'pop hors ménages': 'Pop_Hors_Menages',
# #         'année': 'Annee'
# #     }
    
# #     # Renommer seulement les colonnes qui existent
# #     existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
# #     df = df.rename(columns=existing_mapping)
    
# #     print(f"   ✅ {len(existing_mapping)} colonnes renommées")
    
# #     # === SÉLECTION DES COLONNES ESSENTIELLES ===
# #     print("📊 Sélection des colonnes essentielles...")
    
# #     # Colonnes essentielles pour l'analyse
# #     colonnes_essentielles = [
# #         'ID_Commune',
# #         'Annee', 
# #         'Population_Totale',
# #         'Pop_0_19',
# #         'Pop_20_64', 
# #         'Pop_65_Plus',
# #         'Agriculteurs',
# #         'Cadres_Prof',
# #         'Employes',
# #         'Ouvriers',
# #         'Pop_Francaise',
# #         'Pop_Etrangere',
# #         'Pop_Immigree'
# #     ]
    
# #     # Garder seulement les colonnes qui existent
# #     colonnes_disponibles = [col for col in colonnes_essentielles if col in df.columns]
# #     colonnes_manquantes = [col for col in colonnes_essentielles if col not in df.columns]
    
# #     if colonnes_manquantes:
# #         print(f"   ⚠️  Colonnes manquantes: {colonnes_manquantes}")
    
# #     print(f"   ✅ {len(colonnes_disponibles)} colonnes sélectionnées")
    
# #     # Créer le dataset nettoyé
# #     df_nettoye = df[colonnes_disponibles].copy()
    
# #     # === CRÉATION DE VARIABLES DÉRIVÉES ===
# #     print("🔧 Création de variables dérivées...")
    
# #     if 'Population_Totale' in df_nettoye.columns:
# #         # Pourcentages par groupe d'âge
# #         if 'Pop_0_19' in df_nettoye.columns:
# #             df_nettoye['Pct_0_19'] = (df_nettoye['Pop_0_19'] / df_nettoye['Population_Totale'] * 100).round(2)
        
# #         if 'Pop_20_64' in df_nettoye.columns:
# #             df_nettoye['Pct_20_64'] = (df_nettoye['Pop_20_64'] / df_nettoye['Population_Totale'] * 100).round(2)
        
# #         if 'Pop_65_Plus' in df_nettoye.columns:
# #             df_nettoye['Pct_65_Plus'] = (df_nettoye['Pop_65_Plus'] / df_nettoye['Population_Totale'] * 100).round(2)
        
# #         # Indice de vieillissement
# #         if 'Pop_65_Plus' in df_nettoye.columns and 'Pop_0_19' in df_nettoye.columns:
# #             df_nettoye['Indice_Vieillissement'] = (df_nettoye['Pop_65_Plus'] / (df_nettoye['Pop_0_19'] + 1)).round(2)
        
# #         # Pourcentage population étrangère
# #         if 'Pop_Etrangere' in df_nettoye.columns:
# #             df_nettoye['Pct_Etrangere'] = (df_nettoye['Pop_Etrangere'] / df_nettoye['Population_Totale'] * 100).round(2)
        
# #         # Catégorie de taille commune
# #         df_nettoye['Taille_Commune'] = pd.cut(
# #             df_nettoye['Population_Totale'],
# #             bins=[0, 500, 2000, 10000, 50000, float('inf')],
# #             labels=['Très_Petite', 'Petite', 'Moyenne', 'Grande', 'Très_Grande']
# #         )
        
# #         print("   ✅ Variables dérivées créées")
    
# #     # === CONTRÔLE QUALITÉ ===
# #     print("🔍 Contrôle qualité...")
    
# #     # Valeurs manquantes
# #     valeurs_manquantes = df_nettoye.isnull().sum()
# #     nb_manquantes = valeurs_manquantes.sum()
    
# #     if nb_manquantes > 0:
# #         print(f"   ⚠️  {nb_manquantes} valeurs manquantes détectées")
# #         # Remplacer par 0 pour les colonnes numériques
# #         colonnes_numeriques = df_nettoye.select_dtypes(include=[np.number]).columns
# #         for col in colonnes_numeriques:
# #             if df_nettoye[col].isnull().any():
# #                 df_nettoye[col].fillna(0, inplace=True)
# #         print("   ✅ Valeurs manquantes remplacées par 0")
# #     else:
# #         print("   ✅ Aucune valeur manquante")
    
# #     # Doublons
# #     nb_doublons = df_nettoye.duplicated().sum()
# #     if nb_doublons > 0:
# #         df_nettoye = df_nettoye.drop_duplicates()
# #         print(f"   ✅ {nb_doublons} doublons supprimés")
# #     else:
# #         print("   ✅ Aucun doublon")
    
# #     # Valeurs négatives (ne devrait pas y en avoir pour population)
# #     colonnes_pop = [col for col in df_nettoye.columns if 'Pop' in col or 'Population' in col]
# #     valeurs_negatives = False
# #     for col in colonnes_pop:
# #         if (df_nettoye[col] < 0).any():
# #             valeurs_negatives = True
# #             df_nettoye[col] = df_nettoye[col].clip(lower=0)
    
# #     if valeurs_negatives:
# #         print("   ✅ Valeurs négatives corrigées")
# #     else:
# #         print("   ✅ Aucune valeur négative")
    
# #     # === RÉORGANISATION DES COLONNES ===
# #     print("📋 Réorganisation des colonnes...")
    
# #     # Ordre des colonnes pour l'export
# #     ordre_colonnes = [
# #         'ID_Commune',
# #         'Annee',
# #         'Population_Totale',
# #         'Taille_Commune',
# #         'Pop_0_19',
# #         'Pop_20_64', 
# #         'Pop_65_Plus',
# #         'Pct_0_19',
# #         'Pct_20_64',
# #         'Pct_65_Plus',
# #         'Indice_Vieillissement',
# #         'Agriculteurs',
# #         'Cadres_Prof',
# #         'Employes',
# #         'Ouvriers',
# #         'Pop_Francaise',
# #         'Pop_Etrangere',
# #         'Pct_Etrangere',
# #         'Pop_Immigree'
# #     ]
    
# #     # Garder seulement les colonnes présentes
# #     colonnes_finales = [col for col in ordre_colonnes if col in df_nettoye.columns]
# #     df_final = df_nettoye[colonnes_finales].copy()
    
# #     print(f"   ✅ Dataset final: {df_final.shape[0]} lignes × {df_final.shape[1]} colonnes")
    
# #     return df_final

# # def sauvegarder_dataset(df, nom_base, dossier):
# #     """
# #     Sauvegarde le dataset en Excel et CSV
# #     """
# #     os.makedirs(dossier, exist_ok=True)
    
# #     chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
# #     chemin_csv = os.path.join(dossier, nom_base + ".csv")
    
# #     # Sauvegarde Excel
# #     df.to_excel(chemin_xlsx, index=False, engine='openpyxl')
    
# #     # Sauvegarde CSV
# #     df.to_csv(chemin_csv, index=False, sep=';', encoding='utf-8-sig')
    
# #     return chemin_xlsx, chemin_csv

# # def main():
# #     """
# #     Fonction principale de nettoyage
# #     """
# #     # Nettoyer le dataset
# #     df_nettoye = nettoyer_dataset_population(fichier_population, dossier_sortie)
    
# #     if df_nettoye is None:
# #         return
    
# #     # Sauvegarder
# #     print("\n💾 Sauvegarde du dataset nettoyé...")
# #     nom_fichier = "DATA_Nettoyer_Population"
# #     chemin_excel, chemin_csv = sauvegarder_dataset(df_nettoye, nom_fichier, dossier_sortie)
    
# #     # === RAPPORT FINAL ===
# #     print("\n📋 RAPPORT FINAL")
# #     print("=" * 50)
# #     print(f"✅ Nettoyage réussi")
# #     print(f"📄 Excel : {chemin_excel}")
# #     print(f"📄 CSV   : {chemin_csv}")
# #     print(f"📊 Lignes : {len(df_nettoye):,}")
# #     print(f"📊 Colonnes : {len(df_nettoye.columns)}")
    
# #     if 'Annee' in df_nettoye.columns:
# #         print(f"🗓️  Années : {sorted(df_nettoye['Annee'].unique())}")
    
# #     if 'ID_Commune' in df_nettoye.columns:
# #         print(f"🏛️  Communes : {df_nettoye['ID_Commune'].nunique():,}")
    
# #     if 'Population_Totale' in df_nettoye.columns:
# #         print(f"👥 Population totale : {df_nettoye['Population_Totale'].sum():,}")
# #         print(f"👥 Population moyenne : {df_nettoye['Population_Totale'].mean():,.0f}")
    
# #     print("\n📊 COLONNES FINALES :")
# #     for i, col in enumerate(df_nettoye.columns, 1):
# #         print(f"   {i:2d}. {col}")
    
# #     print("=" * 50)
    
# #     # Aperçu des données
# #     print("\n📋 APERÇU DES DONNÉES :")
# #     print(df_nettoye.head())
    
# #     return df_nettoye

# # # === EXÉCUTION ===
# # if __name__ == "__main__":
# #     dataset_nettoye = main()u
# import os
# import pandas as pd
# import numpy as np

# # === CONFIGURATION DES CHEMINS ===
# fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
# dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Population"

# print("=== NETTOYAGE OPTIMISÉ DATASET POPULATION ===")
# print("=" * 55)

# def nettoyer_dataset_population_optimise(fichier_entree, dossier_sortie):
#     """
#     Nettoie et optimise le dataset population en regroupant les colonnes
#     pour réduire la complexité tout en gardant l'information essentielle
#     """
    
#     # === CHARGEMENT ===
#     try:
#         print("📂 Chargement du dataset population...")
#         df = pd.read_csv(fichier_entree)
#         print(f"   Dataset original: {df.shape[0]} lignes × {df.shape[1]} colonnes")
#     except FileNotFoundError:
#         print(f"❌ Fichier non trouvé : {fichier_entree}")
#         return None
#     except Exception as e:
#         print(f"❌ Erreur de chargement : {e}")
#         return None
    
#     # === RENOMMAGE DES COLONNES ===
#     print("🔧 Renommage des colonnes...")
    
#     # Mapping simplifié des colonnes
#     column_mapping = {
#         'ID Commune': 'ID_Commune',
#         'pop totale': 'Population_Totale',
#         'pop de 0 à 19ans': 'Pop_0_19',
#         'pop de 20 à 64ans': 'Pop_20_64',
#         'pop 65ans ou plus': 'Pop_65_Plus',
#         'pop 15 ans ou plus Agriculteurs': 'Agriculteurs',
#         'pop 15 ans ou plus Artisans, Comm': 'Artisans_Commercants',
#         'pop 15 ans ou plus Cadres, Prof': 'Cadres_Prof',
#         'pop 15 ans ou plus Prof. intermédiaires': 'Prof_Intermediaires',
#         'pop 15 ans ou plus Employés': 'Employes',
#         'pop 15 ans ou plus Ouvriers': 'Ouvriers',
#         'pop Retraités': 'Retraites',
#         'pop 15 ans ou plus sans activité professionnelle': 'Sans_Activite',
#         'pop française': 'Pop_Francaise',
#         'pop étrangère': 'Pop_Etrangere',
#         'pop immigrée': 'Pop_Immigree',
#         'pop en ménages': 'Pop_Menages',
#         'pop hors ménages': 'Pop_Hors_Menages',
#         'année': 'Annee'
#     }
    
#     # Renommer seulement les colonnes qui existent
#     existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
#     df = df.rename(columns=existing_mapping)
#     print(f"   ✅ {len(existing_mapping)} colonnes renommées")
    
#     # === CRÉATION DU DATASET OPTIMISÉ ===
#     print("🔄 Création du dataset optimisé avec colonnes regroupées...")
    
#     # Initialiser le nouveau dataset avec les colonnes de base
#     df_optimise = pd.DataFrame()
    
#     # 1. IDENTIFIANTS ET INFORMATIONS DE BASE
#     if 'ID_Commune' in df.columns:
#         df_optimise['ID_Commune'] = df['ID_Commune']
#     if 'Annee' in df.columns:
#         df_optimise['Annee'] = df['Annee']
#     if 'Population_Totale' in df.columns:
#         df_optimise['Population_Totale'] = df['Population_Totale']
    
#     # 2. GROUPES D'ÂGE CONSOLIDÉS
#     print("   📊 Regroupement des tranches d'âge...")
    
#     # Population jeune (0-19 ans) - déjà une colonne
#     if 'Pop_0_19' in df.columns:
#         df_optimise['Pop_Jeunes'] = df['Pop_0_19']
#         df_optimise['Pct_Jeunes'] = (df['Pop_0_19'] / df['Population_Totale'] * 100).round(2)
    
#     # Population active (20-64 ans) - déjà une colonne
#     if 'Pop_20_64' in df.columns:
#         df_optimise['Pop_Actifs'] = df['Pop_20_64']
#         df_optimise['Pct_Actifs'] = (df['Pop_20_64'] / df['Population_Totale'] * 100).round(2)
    
#     # Population seniors (65+ ans) - déjà une colonne
#     if 'Pop_65_Plus' in df.columns:
#         df_optimise['Pop_Seniors'] = df['Pop_65_Plus']
#         df_optimise['Pct_Seniors'] = (df['Pop_65_Plus'] / df['Population_Totale'] * 100).round(2)
    
#     # 3. CATÉGORIES SOCIO-PROFESSIONNELLES REGROUPÉES
#     print("   💼 Regroupement des catégories socio-professionnelles...")
    
#     # CSP Supérieures (Agriculteurs + Cadres/Prof + Artisans/Commerçants)
#     colonnes_csp_sup = ['Agriculteurs', 'Cadres_Prof', 'Artisans_Commercants']
#     colonnes_csp_sup_dispo = [col for col in colonnes_csp_sup if col in df.columns]
#     if colonnes_csp_sup_dispo:
#         df_optimise['CSP_Superieures'] = df[colonnes_csp_sup_dispo].sum(axis=1)
#         df_optimise['Pct_CSP_Sup'] = (df_optimise['CSP_Superieures'] / df['Population_Totale'] * 100).round(2)
    
#     # CSP Moyennes (Professions intermédiaires + Employés)
#     colonnes_csp_moy = ['Prof_Intermediaires', 'Employes']
#     colonnes_csp_moy_dispo = [col for col in colonnes_csp_moy if col in df.columns]
#     if colonnes_csp_moy_dispo:
#         df_optimise['CSP_Moyennes'] = df[colonnes_csp_moy_dispo].sum(axis=1)
#         df_optimise['Pct_CSP_Moy'] = (df_optimise['CSP_Moyennes'] / df['Population_Totale'] * 100).round(2)
    
#     # CSP Populaires (Ouvriers)
#     if 'Ouvriers' in df.columns:
#         df_optimise['CSP_Populaires'] = df['Ouvriers']
#         df_optimise['Pct_CSP_Pop'] = (df['Ouvriers'] / df['Population_Totale'] * 100).round(2)
    
#     # Population sans activité (Retraités + Sans activité)
#     colonnes_sans_act = ['Retraites', 'Sans_Activite']
#     colonnes_sans_act_dispo = [col for col in colonnes_sans_act if col in df.columns]
#     if colonnes_sans_act_dispo:
#         df_optimise['Pop_Sans_Activite'] = df[colonnes_sans_act_dispo].sum(axis=1)
#         df_optimise['Pct_Sans_Activite'] = (df_optimise['Pop_Sans_Activite'] / df['Population_Totale'] * 100).round(2)
    
#     # 4. ORIGINE ET NATIONALITÉ CONSOLIDÉE
#     print("   🌍 Regroupement des données de nationalité...")
    
#     # Population française (déjà consolidée)
#     if 'Pop_Francaise' in df.columns:
#         df_optimise['Pop_Francaise'] = df['Pop_Francaise']
#         df_optimise['Pct_Francaise'] = (df['Pop_Francaise'] / df['Population_Totale'] * 100).round(2)
    
#     # Population étrangère/immigrée (regroupement)
#     colonnes_etrangere = ['Pop_Etrangere', 'Pop_Immigree']
#     colonnes_etrangere_dispo = [col for col in colonnes_etrangere if col in df.columns]
#     if colonnes_etrangere_dispo:
#         # Prendre le maximum des deux (car Pop_Immigree inclut souvent Pop_Etrangere)
#         df_optimise['Pop_Etrangere_Immigree'] = df[colonnes_etrangere_dispo].max(axis=1)
#         df_optimise['Pct_Etrangere_Immigree'] = (df_optimise['Pop_Etrangere_Immigree'] / df['Population_Totale'] * 100).round(2)
    
#     # 5. STRUCTURE DES MÉNAGES CONSOLIDÉE
#     print("   🏠 Regroupement des données de ménages...")
    
#     # Ratio population en ménages vs hors ménages
#     if 'Pop_Menages' in df.columns and 'Pop_Hors_Menages' in df.columns:
#         df_optimise['Pop_En_Menages'] = df['Pop_Menages']
#         df_optimise['Pct_En_Menages'] = (df['Pop_Menages'] / df['Population_Totale'] * 100).round(2)
#     elif 'Pop_Menages' in df.columns:
#         df_optimise['Pop_En_Menages'] = df['Pop_Menages']
#         df_optimise['Pct_En_Menages'] = (df['Pop_Menages'] / df['Population_Totale'] * 100).round(2)
    
#     # 6. INDICATEURS SYNTHÉTIQUES
#     print("   📈 Création d'indicateurs synthétiques...")
    
#     # Indice de vieillissement
#     if 'Pop_Seniors' in df_optimise.columns and 'Pop_Jeunes' in df_optimise.columns:
#         df_optimise['Indice_Vieillissement'] = (df_optimise['Pop_Seniors'] / (df_optimise['Pop_Jeunes'] + 1)).round(2)
    
#     # Indice de diversité sociale (équilibre entre CSP)
#     if all(col in df_optimise.columns for col in ['CSP_Superieures', 'CSP_Moyennes', 'CSP_Populaires']):
#         total_csp = df_optimise['CSP_Superieures'] + df_optimise['CSP_Moyennes'] + df_optimise['CSP_Populaires']
#         # Calcul d'un indice de diversité (plus proche de 1 = plus équilibré)
#         df_optimise['Diversite_Sociale'] = (1 - abs(df_optimise['CSP_Superieures'] - df_optimise['CSP_Moyennes'] - df_optimise['CSP_Populaires']) / (total_csp + 1)).round(3)
    
#     # Catégorie de taille commune
#     if 'Population_Totale' in df_optimise.columns:
#         df_optimise['Categorie_Commune'] = pd.cut(
#             df_optimise['Population_Totale'],
#             bins=[0, 500, 2000, 10000, 50000, float('inf')],
#             labels=['Rural', 'Bourg', 'Petite_Ville', 'Ville_Moyenne', 'Grande_Ville']
#         )
    
#     # === CONTRÔLE QUALITÉ ===
#     print("🔍 Contrôle qualité du dataset optimisé...")
    
#     # Valeurs manquantes
#     valeurs_manquantes = df_optimise.isnull().sum()
#     nb_manquantes = valeurs_manquantes.sum()
    
#     if nb_manquantes > 0:
#         print(f"   ⚠️  {nb_manquantes} valeurs manquantes détectées")
#         # Remplacer par 0 pour les colonnes numériques (sauf catégorielles)
#         colonnes_numeriques = df_optimise.select_dtypes(include=[np.number]).columns
#         for col in colonnes_numeriques:
#             if df_optimise[col].isnull().any():
#                 df_optimise[col].fillna(0, inplace=True)
#         print("   ✅ Valeurs manquantes remplacées par 0")
#     else:
#         print("   ✅ Aucune valeur manquante")
    
#     # Doublons
#     nb_doublons = df_optimise.duplicated().sum()
#     if nb_doublons > 0:
#         df_optimise = df_optimise.drop_duplicates()
#         print(f"   ✅ {nb_doublons} doublons supprimés")
#     else:
#         print("   ✅ Aucun doublon")
    
#     # Valeurs négatives
#     colonnes_pop = [col for col in df_optimise.columns if 'Pop' in col or 'CSP' in col]
#     valeurs_negatives = False
#     for col in colonnes_pop:
#         if col in df_optimise.columns and (df_optimise[col] < 0).any():
#             valeurs_negatives = True
#             df_optimise[col] = df_optimise[col].clip(lower=0)
    
#     if valeurs_negatives:
#         print("   ✅ Valeurs négatives corrigées")
#     else:
#         print("   ✅ Aucune valeur négative")
    
#     print(f"   ✅ Dataset optimisé: {df_optimise.shape[0]} lignes × {df_optimise.shape[1]} colonnes")
#     print(f"   📉 Réduction: {df.shape[1]} → {df_optimise.shape[1]} colonnes ({df.shape[1] - df_optimise.shape[1]} colonnes en moins)")
    
#     return df_optimise

# def sauvegarder_dataset(df, nom_base, dossier):
#     """
#     Sauvegarde le dataset en Excel et CSV
#     """
#     os.makedirs(dossier, exist_ok=True)
    
#     chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
#     chemin_csv = os.path.join(dossier, nom_base + ".csv")
    
#     # Sauvegarde Excel avec formatage
#     with pd.ExcelWriter(chemin_xlsx, engine='openpyxl') as writer:
#         df.to_excel(writer, sheet_name='Population_Optimisee', index=False)
    
#     # Sauvegarde CSV
#     df.to_csv(chemin_csv, index=False, sep=';', encoding='utf-8-sig')
    
#     return chemin_xlsx, chemin_csv

# def generer_rapport_optimisation(df_original, df_optimise):
#     """
#     Génère un rapport détaillé de l'optimisation
#     """
#     print("\n📊 RAPPORT D'OPTIMISATION")
#     print("=" * 50)
    
#     print(f"🔢 Colonnes originales : {df_original.shape[1]}")
#     print(f"🔢 Colonnes optimisées : {df_optimise.shape[1]}")
#     print(f"📉 Réduction : {df_original.shape[1] - df_optimise.shape[1]} colonnes (-{((df_original.shape[1] - df_optimise.shape[1]) / df_original.shape[1] * 100):.1f}%)")
    
#     print("\n📋 NOUVELLES COLONNES CRÉÉES :")
#     nouvelles_colonnes = [
#         "Pop_Jeunes, Pop_Actifs, Pop_Seniors (+ pourcentages)",
#         "CSP_Superieures (Agriculteurs + Cadres + Artisans)",
#         "CSP_Moyennes (Prof. intermédiaires + Employés)",
#         "CSP_Populaires (Ouvriers)",
#         "Pop_Sans_Activite (Retraités + Sans activité)",
#         "Pop_Etrangere_Immigree (consolidation)",
#         "Indice_Vieillissement",
#         "Diversite_Sociale",
#         "Categorie_Commune"
#     ]
    
#     for i, col in enumerate(nouvelles_colonnes, 1):
#         print(f"   {i}. {col}")

# def main():
#     """
#     Fonction principale de nettoyage optimisé
#     """
#     # Charger le dataset original pour comparaison
#     try:
#         df_original = pd.read_csv(fichier_population)
#     except:
#         df_original = None
    
#     # Nettoyer et optimiser le dataset
#     df_optimise = nettoyer_dataset_population_optimise(fichier_population, dossier_sortie)
    
#     if df_optimise is None:
#         return
    
#     # Sauvegarder
#     print("\n💾 Sauvegarde du dataset optimisé...")
#     nom_fichier = "DATA_Population_Optimise"
#     chemin_excel, chemin_csv = sauvegarder_dataset(df_optimise, nom_fichier, dossier_sortie)
    
#     # Rapport d'optimisation
#     if df_original is not None:
#         generer_rapport_optimisation(df_original, df_optimise)
    
#     # === RAPPORT FINAL ===
#     print("\n📋 RAPPORT FINAL")
#     print("=" * 50)
#     print(f"✅ Optimisation réussie")
#     print(f"📄 Excel : {chemin_excel}")
#     print(f"📄 CSV   : {chemin_csv}")
#     print(f"📊 Lignes : {len(df_optimise):,}")
#     print(f"📊 Colonnes : {len(df_optimise.columns)}")
    
#     if 'Annee' in df_optimise.columns:
#         print(f"🗓️  Années : {sorted(df_optimise['Annee'].unique())}")
    
#     if 'ID_Commune' in df_optimise.columns:
#         print(f"🏛️  Communes : {df_optimise['ID_Commune'].nunique():,}")
    
#     if 'Population_Totale' in df_optimise.columns:
#         print(f"👥 Population totale : {df_optimise['Population_Totale'].sum():,}")
#         print(f"👥 Population moyenne : {df_optimise['Population_Totale'].mean():,.0f}")
    
#     print("\n📊 COLONNES FINALES OPTIMISÉES :")
#     for i, col in enumerate(df_optimise.columns, 1):
#         print(f"   {i:2d}. {col}")
    
#     print("=" * 50)
    
#     # Aperçu des données
#     print("\n📋 APERÇU DES DONNÉES OPTIMISÉES :")
#     print(df_optimise.head(3))
    
#     # Statistiques descriptives des nouvelles colonnes
#     if len(df_optimise) > 0:
#         print("\n📈 STATISTIQUES DES INDICATEURS SYNTHÉTIQUES :")
#         colonnes_stats = ['Indice_Vieillissement', 'Pct_Seniors', 'Pct_CSP_Sup', 'Pct_Etrangere_Immigree']
#         colonnes_stats_dispo = [col for col in colonnes_stats if col in df_optimise.columns]
        
#         if colonnes_stats_dispo:
#             print(df_optimise[colonnes_stats_dispo].describe().round(2))
    
#     return df_optimise

# # === EXÉCUTION ===
# if __name__ == "__main__":
#     dataset_optimise = main()
import os
import pandas as pd
import numpy as np

# === CONFIGURATION DES CHEMINS ===
fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/population_csv.csv"
dossier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_Population"

print("=== DATASET POPULATION OPTIMISÉ POUR PRÉDICTIONS ===")
print("=" * 60)

def creer_dataset_prediction(fichier_entree, dossier_sortie):
    """
    Crée un dataset optimisé pour les prédictions en éliminant la redondance
    et en choisissant les meilleurs formats (pourcentages vs nombres absolus)
    """
    
    # === CHARGEMENT ===
    try:
        print("📂 Chargement du dataset population...")
        df = pd.read_csv(fichier_entree)
        print(f"   Dataset original: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {fichier_entree}")
        return None
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return None
    
    # === RENOMMAGE DES COLONNES ===
    print("🔧 Renommage des colonnes...")
    
    column_mapping = {
        'ID Commune': 'ID_Commune',
        'pop totale': 'Population_Totale',
        'pop de 0 à 19ans': 'Pop_0_19',
        'pop de 20 à 64ans': 'Pop_20_64',
        'pop 65ans ou plus': 'Pop_65_Plus',
        'pop 15 ans ou plus Agriculteurs': 'Agriculteurs',
        'pop 15 ans ou plus Artisans, Comm': 'Artisans_Commercants',
        'pop 15 ans ou plus Cadres, Prof': 'Cadres_Prof',
        'pop 15 ans ou plus Prof. intermédiaires': 'Prof_Intermediaires',
        'pop 15 ans ou plus Employés': 'Employes',
        'pop 15 ans ou plus Ouvriers': 'Ouvriers',
        'pop Retraités': 'Retraites',
        'pop 15 ans ou plus sans activité professionnelle': 'Sans_Activite',
        'pop française': 'Pop_Francaise',
        'pop étrangère': 'Pop_Etrangere',
        'pop immigrée': 'Pop_Immigree',
        'pop en ménages': 'Pop_Menages',
        'pop hors ménages': 'Pop_Hors_Menages',
        'année': 'Annee'
    }
    
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mapping)
    print(f"   ✅ {len(existing_mapping)} colonnes renommées")
    
    # === CRÉATION DU DATASET POUR PRÉDICTIONS ===
    print("🎯 Création du dataset optimisé pour prédictions...")
    print("   📊 Choix stratégique: POURCENTAGES pour normalisation et comparaison")
    print("   📊 NOMBRES ABSOLUS uniquement pour variables de référence")
    
    df_pred = pd.DataFrame()
    
    # === 1. VARIABLES DE RÉFÉRENCE (NOMBRES ABSOLUS) ===
    print("\n🔑 Variables de référence (nombres absolus)...")
    
    # Identifiants essentiels
    if 'ID_Commune' in df.columns:
        df_pred['ID_Commune'] = df['ID_Commune']
    if 'Annee' in df.columns:
        df_pred['Annee'] = df['Annee']
    
    # Population totale (référence pour calculs et taille)
    if 'Population_Totale' in df.columns:
        df_pred['Population_Totale'] = df['Population_Totale']
        print("   ✅ Population_Totale conservée (référence de taille)")
    
    # === 2. STRUCTURE DÉMOGRAPHIQUE (POURCENTAGES) ===
    print("\n👥 Structure démographique (pourcentages)...")
    
    # Pourcentages par groupe d'âge (meilleur pour prédictions démographiques)
    if all(col in df.columns for col in ['Pop_0_19', 'Pop_20_64', 'Pop_65_Plus', 'Population_Totale']):
        df_pred['Pct_Jeunes'] = (df['Pop_0_19'] / df['Population_Totale'] * 100).round(2)
        df_pred['Pct_Actifs'] = (df['Pop_20_64'] / df['Population_Totale'] * 100).round(2)
        df_pred['Pct_Seniors'] = (df['Pop_65_Plus'] / df['Population_Totale'] * 100).round(2)
        print("   ✅ Structure par âge en pourcentages")
    
    # === 3. STRUCTURE SOCIO-PROFESSIONNELLE (POURCENTAGES) ===
    print("\n💼 Structure socio-professionnelle (pourcentages)...")
    
    # Regroupement et calcul en pourcentages pour analyse comparative
    colonnes_csp = ['Agriculteurs', 'Cadres_Prof', 'Artisans_Commercants', 
                   'Prof_Intermediaires', 'Employes', 'Ouvriers']
    colonnes_csp_dispo = [col for col in colonnes_csp if col in df.columns]
    
    if len(colonnes_csp_dispo) >= 3:
        # CSP Supérieures en pourcentage
        csp_sup_cols = ['Agriculteurs', 'Cadres_Prof', 'Artisans_Commercants']
        csp_sup_dispo = [col for col in csp_sup_cols if col in df.columns]
        if csp_sup_dispo:
            csp_sup_total = df[csp_sup_dispo].sum(axis=1)
            df_pred['Pct_CSP_Superieures'] = (csp_sup_total / df['Population_Totale'] * 100).round(2)
        
        # CSP Moyennes en pourcentage
        csp_moy_cols = ['Prof_Intermediaires', 'Employes']
        csp_moy_dispo = [col for col in csp_moy_cols if col in df.columns]
        if csp_moy_dispo:
            csp_moy_total = df[csp_moy_dispo].sum(axis=1)
            df_pred['Pct_CSP_Moyennes'] = (csp_moy_total / df['Population_Totale'] * 100).round(2)
        
        # CSP Populaires en pourcentage
        if 'Ouvriers' in df.columns:
            df_pred['Pct_CSP_Populaires'] = (df['Ouvriers'] / df['Population_Totale'] * 100).round(2)
        
        print("   ✅ Structure CSP en pourcentages")
    
    # === 4. ACTIVITÉ ÉCONOMIQUE (POURCENTAGES) ===
    print("\n📈 Activité économique (pourcentages)...")
    
    # Population sans activité professionnelle (retraités + sans activité)
    colonnes_sans_act = ['Retraites', 'Sans_Activite']
    colonnes_sans_act_dispo = [col for col in colonnes_sans_act if col in df.columns]
    if colonnes_sans_act_dispo:
        sans_act_total = df[colonnes_sans_act_dispo].sum(axis=1)
        df_pred['Pct_Sans_Activite'] = (sans_act_total / df['Population_Totale'] * 100).round(2)
        print("   ✅ Population sans activité en pourcentage")
    
    # === 5. DIVERSITÉ ET ORIGINE (POURCENTAGES) ===
    print("\n🌍 Diversité et origine (pourcentages)...")
    
    # Pourcentage population française
    if 'Pop_Francaise' in df.columns:
        df_pred['Pct_Francaise'] = (df['Pop_Francaise'] / df['Population_Totale'] * 100).round(2)
    
    # Pourcentage population étrangère/immigrée (prendre le max des deux)
    colonnes_etrangere = ['Pop_Etrangere', 'Pop_Immigree']
    colonnes_etrangere_dispo = [col for col in colonnes_etrangere if col in df.columns]
    if colonnes_etrangere_dispo:
        pop_etrangere_max = df[colonnes_etrangere_dispo].max(axis=1)
        df_pred['Pct_Etrangere'] = (pop_etrangere_max / df['Population_Totale'] * 100).round(2)
        print("   ✅ Diversité d'origine en pourcentages")
    
    # === 6. STRUCTURE DES MÉNAGES (POURCENTAGES) ===
    print("\n🏠 Structure des ménages (pourcentages)...")
    
    if 'Pop_Menages' in df.columns:
        df_pred['Pct_En_Menages'] = (df['Pop_Menages'] / df['Population_Totale'] * 100).round(2)
        print("   ✅ Structure ménages en pourcentage")
    
    # === 7. INDICATEURS SYNTHÉTIQUES POUR PRÉDICTIONS ===
    print("\n🎯 Indicateurs synthétiques pour prédictions...")
    
    # Indice de vieillissement (ratio crucial pour prédictions démographiques)
    if 'Pct_Seniors' in df_pred.columns and 'Pct_Jeunes' in df_pred.columns:
        # Éviter division par zéro
        df_pred['Indice_Vieillissement'] = (df_pred['Pct_Seniors'] / (df_pred['Pct_Jeunes'] + 0.1)).round(3)
        print("   ✅ Indice de vieillissement")
    
    # Indice de diversité sociale (équilibre CSP)
    if all(col in df_pred.columns for col in ['Pct_CSP_Superieures', 'Pct_CSP_Moyennes', 'Pct_CSP_Populaires']):
        # Calcul de l'entropie sociale (plus élevé = plus diversifié)
        total_csp = df_pred['Pct_CSP_Superieures'] + df_pred['Pct_CSP_Moyennes'] + df_pred['Pct_CSP_Populaires']
        total_csp = total_csp.replace(0, 0.1)  # Éviter division par zéro
        
        p1 = df_pred['Pct_CSP_Superieures'] / total_csp
        p2 = df_pred['Pct_CSP_Moyennes'] / total_csp
        p3 = df_pred['Pct_CSP_Populaires'] / total_csp
        
        # Entropie de Shannon adaptée
        entropy = -(p1 * np.log(p1 + 0.001) + p2 * np.log(p2 + 0.001) + p3 * np.log(p3 + 0.001))
        df_pred['Diversite_Sociale'] = (entropy / np.log(3)).round(3)  # Normalisé entre 0 et 1
        print("   ✅ Indice de diversité sociale")
    
    # Indice de dépendance démographique
    if all(col in df_pred.columns for col in ['Pct_Jeunes', 'Pct_Seniors', 'Pct_Actifs']):
        df_pred['Taux_Dependance'] = ((df_pred['Pct_Jeunes'] + df_pred['Pct_Seniors']) / df_pred['Pct_Actifs']).round(3)
        print("   ✅ Taux de dépendance démographique")
    
    # === 8. VARIABLES CATÉGORIELLES POUR ML ===
    print("\n🤖 Variables catégorielles pour Machine Learning...")
    
    # Catégorie de taille (utile pour segmentation)
    if 'Population_Totale' in df_pred.columns:
        df_pred['Taille_Commune'] = pd.cut(
            df_pred['Population_Totale'],
            bins=[0, 500, 2000, 10000, 50000, float('inf')],
            labels=[1, 2, 3, 4, 5]  # Encodage numérique pour ML
        ).astype(int)
        print("   ✅ Catégorie de taille (encodage numérique)")
    
    # Profil démographique dominant
    if all(col in df_pred.columns for col in ['Pct_Jeunes', 'Pct_Actifs', 'Pct_Seniors']):
        conditions = [
            df_pred['Pct_Jeunes'] > df_pred[['Pct_Actifs', 'Pct_Seniors']].max(axis=1),
            df_pred['Pct_Actifs'] > df_pred[['Pct_Jeunes', 'Pct_Seniors']].max(axis=1),
            df_pred['Pct_Seniors'] > df_pred[['Pct_Jeunes', 'Pct_Actifs']].max(axis=1)
        ]
        choices = [1, 2, 3]  # 1=Jeune, 2=Actif, 3=Vieillissant
        df_pred['Profil_Demographique'] = np.select(conditions, choices, default=2)
        print("   ✅ Profil démographique dominant")
    
    # === CONTRÔLE QUALITÉ ===
    print("\n🔍 Contrôle qualité final...")
    
    # Valeurs manquantes
    valeurs_manquantes = df_pred.isnull().sum().sum()
    if valeurs_manquantes > 0:
        # Remplacer par médiane pour pourcentages
        for col in df_pred.columns:
            if df_pred[col].dtype in ['float64', 'int64'] and df_pred[col].isnull().any():
                if 'Pct_' in col or 'Indice_' in col or 'Taux_' in col:
                    df_pred[col].fillna(df_pred[col].median(), inplace=True)
                else:
                    df_pred[col].fillna(0, inplace=True)
        print(f"   ✅ {valeurs_manquantes} valeurs manquantes corrigées")
    
    # Valeurs aberrantes pour pourcentages
    colonnes_pct = [col for col in df_pred.columns if 'Pct_' in col]
    for col in colonnes_pct:
        df_pred[col] = df_pred[col].clip(0, 100)
    
    # Doublons
    nb_doublons = df_pred.duplicated().sum()
    if nb_doublons > 0:
        df_pred = df_pred.drop_duplicates()
        print(f"   ✅ {nb_doublons} doublons supprimés")
    
    print(f"\n✅ Dataset final pour prédictions: {df_pred.shape[0]} lignes × {df_pred.shape[1]} colonnes")
        # === FILTRAGE FINAL DES COLONNES POUR PRÉDICTION ÉLECTORALE ===
    print("\n🧹 Filtrage final : conservation des variables pertinentes pour les élections...")

    colonnes_utiles = [
        'ID_Commune', 'Annee',  # utiles pour suivi
        'Population_Totale',
        'Pct_Jeunes',
        'Pct_Seniors',
        'Pct_Sans_Activite',
        'Pct_Etrangere',
    ]
    colonnes_existantes = [col for col in colonnes_utiles if col in df_pred.columns]
    df_pred = df_pred[colonnes_existantes]

    print(f"   ✅ Variables conservées : {len(colonnes_existantes)}")

    return df_pred



def analyser_correlations(df):
    """
    Analyse les corrélations pour identifier les variables les plus prédictives
    """
    print("\n🔬 ANALYSE DES CORRÉLATIONS POUR PRÉDICTIONS")
    print("=" * 50)
    
    # Sélectionner les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
    colonnes_numeriques = [col for col in colonnes_numeriques if col not in ['ID_Commune', 'Annee']]
    
    if len(colonnes_numeriques) > 1:
        correlation_matrix = df[colonnes_numeriques].corr()
        
        # Identifier les corrélations fortes (>0.7 ou <-0.7)
        correlations_fortes = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    correlations_fortes.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        if correlations_fortes:
            print("⚠️  CORRÉLATIONS FORTES DÉTECTÉES :")
            for corr in correlations_fortes:
                print(f"   {corr['var1']} ↔ {corr['var2']}: {corr['correlation']}")
        else:
            print("✅ Pas de corrélations trop fortes (>0.7)")
        
        # Variables les plus variables (bonnes pour prédictions)
        variances = df[colonnes_numeriques].var().sort_values(ascending=False)
        print(f"\n📊 TOP 5 VARIABLES LES PLUS VARIABLES :")
        for i, (var, val) in enumerate(variances.head().items(), 1):
            print(f"   {i}. {var}: {val:.2f}")

def sauvegarder_dataset_prediction(df, nom_base, dossier):
    """
    Sauvegarde optimisée pour datasets de prédiction
    """
    os.makedirs(dossier, exist_ok=True)
    
    chemin_xlsx = os.path.join(dossier, nom_base + ".xlsx")
    chemin_csv = os.path.join(dossier, nom_base + ".csv")
    
    # Sauvegarde Excel avec métadonnées
    with pd.ExcelWriter(chemin_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data_Prediction', index=False)
        
        # Feuille avec description des variables
        descriptions = pd.DataFrame({
            'Variable': df.columns,
            'Type': ['Pourcentage' if 'Pct_' in col or 'Taux_' in col 
                    else 'Indice' if 'Indice_' in col or 'Diversite_' in col
                    else 'Numérique' for col in df.columns],
            'Usage': ['Prédiction' for _ in df.columns]
        })
        descriptions.to_excel(writer, sheet_name='Description_Variables', index=False)
    
    # Sauvegarde CSV optimisé pour ML
    df.to_csv(chemin_csv, index=False, sep=',', encoding='utf-8')  # Virgule pour ML
    
    return chemin_xlsx, chemin_csv

def main():
    """
    Fonction principale - Dataset optimisé pour prédictions
    """
    # Créer le dataset pour prédictions
    df_prediction = creer_dataset_prediction(fichier_population, dossier_sortie)
    
    if df_prediction is None:
        return
    
    # Analyser les corrélations
    analyser_correlations(df_prediction)
    
    # Sauvegarder
    print("\n💾 Sauvegarde du dataset pour population...")
    nom_fichier = "DATA_Population_finale"
    chemin_excel, chemin_csv = sauvegarder_dataset_prediction(df_prediction, nom_fichier, dossier_sortie)
    
    # === RAPPORT FINAL ===
    print("\n📋 RAPPORT FINAL - DATASET PRÉDICTIONS")
    print("=" * 60)
    print(f"✅ Dataset optimisé pour Machine Learning créé")
    print(f"📄 Excel : {chemin_excel}")
    print(f"📄 CSV   : {chemin_csv}")
    print(f"📊 Lignes : {len(df_prediction):,}")
    print(f"📊 Colonnes : {len(df_prediction.columns)}")
    
    print(f"\n🎯 OPTIMISATIONS POUR PRÉDICTIONS :")
    print(f"   • POURCENTAGES prioritaires (normalisation)")
    print(f"   • Élimination de la redondance nombres/pourcentages")
    print(f"   • Indicateurs synthétiques calculés")
    print(f"   • Variables catégorielles encodées numériquement")
    print(f"   • Corrélations analysées")
    
    if 'Annee' in df_prediction.columns:
        print(f"\n🗓️  Années disponibles : {sorted(df_prediction['Annee'].unique())}")
    
    if 'ID_Commune' in df_prediction.columns:
        print(f"🏛️  Communes : {df_prediction['ID_Commune'].nunique():,}")
    
    print(f"\n📊 VARIABLES FINALES POUR PRÉDICTIONS :")
    variables_par_type = {
        'Référence': [col for col in df_prediction.columns if col in ['ID_Commune', 'Annee', 'Population_Totale']],
        'Pourcentages': [col for col in df_prediction.columns if 'Pct_' in col],
        'Indicateurs': [col for col in df_prediction.columns if 'Indice_' in col or 'Taux_' in col or 'Diversite_' in col],
        'Catégorielles': [col for col in df_prediction.columns if col in ['Taille_Commune', 'Profil_Demographique']]
    }
    
    for type_var, variables in variables_par_type.items():
        if variables:
            print(f"\n   {type_var} ({len(variables)}) :")
            for var in variables:
                print(f"     • {var}")
    
    print("=" * 60)
    
    # Aperçu final
    print(f"\n📋 APERÇU DES DONNÉES :")
    print(df_prediction.head(3))
    
    print(f"\n📈 STATISTIQUES DESCRIPTIVES :")
    colonnes_stats = [col for col in df_prediction.columns if 'Pct_' in col or 'Indice_' in col][:5]
    if colonnes_stats:
        print(df_prediction[colonnes_stats].describe().round(2))
    
    return df_prediction

# === EXÉCUTION ===
if __name__ == "__main__":
    dataset_prediction = main()