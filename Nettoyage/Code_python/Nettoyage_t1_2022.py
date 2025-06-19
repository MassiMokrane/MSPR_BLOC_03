# import os
# import pandas as pd
# from collections import Counter

# # === 1. Charger le fichier Excel ===
# fichier_source = "C:/Users/Massi/Desktop/MSPR BLOC 03/DATA_Set_brut/2022_tour1.xlsx"
# feuille = "commune"
# df_brut = pd.read_excel(fichier_source, sheet_name=feuille)

# # === 2. Détecter automatiquement la ligne d'en-tête ===
# ligne_entete_index = df_brut[df_brut.eq("Code du département").any(axis=1)].index[0]
# df = df_brut[ligne_entete_index:].reset_index(drop=True)
# df.columns = df.iloc[0]
# df = df[1:].reset_index(drop=True)

# # === 3. Nettoyage des colonnes vides et caractères spéciaux ===
# # NE PAS supprimer les colonnes sans nom pour l'instant, on va les renommer
# df.columns = df.columns.astype(str)  # Convertir en string pour éviter les erreurs
# df.columns = df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True)
# df.columns = df.columns.str.strip()

# print("Colonnes avant renommage :")
# for i, col in enumerate(df.columns):
#     print(f"{i}: '{col}'")
# print("--------------------------------------------------")

# # === 4. Vérifier les colonnes dupliquées ===
# compte_colonnes = Counter(df.columns)
# colonnes_dupliquees = {col: count for col, count in compte_colonnes.items() if count > 1}
# nb_lignes_dupliquees = df.duplicated().sum()

# print(" Analyse du fichier COMMUNE :")
# print("--------------------------------------------------")
# print(f"Nombre total de colonnes         : {len(df.columns)}")
# print(f"Nombre de colonnes uniques       : {len(set(df.columns))}")
# print(f"Nombre de colonnes dupliquées    : {len(colonnes_dupliquees)}")
# print(f"Nombre de lignes dupliquées      : {nb_lignes_dupliquees}")
# print("--------------------------------------------------")

# # === 5. Renommer dynamiquement TOUTES les colonnes candidats ===
# def renommer_colonnes_candidats(colonnes: list) -> list:
#     nouvelles_colonnes = []
#     compteur_candidat = 0
#     sequence_candidat = ["N°Panneau", "Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]
#     position_dans_sequence = 0
#     candidat_commence = False
    
#     for i, col in enumerate(colonnes):
#         # Détecter le début des données candidats
#         if col == "N°Panneau" and not candidat_commence:
#             candidat_commence = True
#             compteur_candidat = 1
#             position_dans_sequence = 0
#             nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
#             position_dans_sequence += 1
#         elif candidat_commence:
#             # Si on est dans une séquence de candidat
#             if col in sequence_candidat or col == "nan" or col.strip() == "" or pd.isna(col):
#                 # Si c'est le début d'un nouveau candidat (N°Panneau)
#                 if position_dans_sequence == 0 or position_dans_sequence >= len(sequence_candidat):
#                     compteur_candidat += 1
#                     position_dans_sequence = 0
                
#                 # Assigner le nom de la colonne selon la position dans la séquence
#                 if position_dans_sequence < len(sequence_candidat):
#                     nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
#                     position_dans_sequence += 1
#                 else:
#                     # Si on dépasse la séquence, on recommence
#                     compteur_candidat += 1
#                     position_dans_sequence = 0
#                     nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
#                     position_dans_sequence += 1
#             else:
#                 # Si on rencontre une colonne qui n'est pas dans la séquence candidat
#                 # on continue la séquence actuelle
#                 if position_dans_sequence < len(sequence_candidat):
#                     nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
#                     position_dans_sequence += 1
#                 else:
#                     nouvelles_colonnes.append(col)
#         else:
#             # Colonnes avant les candidats
#             nouvelles_colonnes.append(col)
    
#     return nouvelles_colonnes

# def appliquer_renommage(df):
#     df.columns = renommer_colonnes_candidats(list(df.columns))
#     return df

# # APPLIQUER LE RENOMMAGE D'ABORD
# df = appliquer_renommage(df)

# print("Colonnes après renommage :")
# for i, col in enumerate(df.columns):
#     print(f"{i}: '{col}'")
# print("--------------------------------------------------")

# # === 6. Créer la colonne ID Commune ===
# df["Code du département"] = df["Code du département"].astype(str).str.zfill(2)
# df["Code de la commune"] = df["Code de la commune"].astype(str).str.zfill(3)
# df["ID Commune"] = df["Code du département"] + df["Code de la commune"]

# # === 7. Identifier l'élu APRÈS le renommage ===
# # Maintenant on récupère TOUTES les colonnes renommées
# colonnes_voix = [col for col in df.columns if col.startswith("Voix ")]
# colonnes_nom = [col for col in df.columns if col.startswith("Nom ")]
# colonnes_prenom = [col for col in df.columns if col.startswith("Prénom ")]
# colonnes_sexe = [col for col in df.columns if col.startswith("Sexe ")]
# colonnes_panneau = [col for col in df.columns if col.startswith("N°Panneau ")]
# colonnes_pourcent_ins = [col for col in df.columns if col.startswith("% Voix/Ins ")]
# colonnes_pourcent_exp = [col for col in df.columns if col.startswith("% Voix/Exp ")]

# print(f"Colonnes de voix trouvées : {colonnes_voix}")  # Debug pour vérifier
# print(f"Colonnes de nom trouvées : {colonnes_nom}")
# print(f"Nombre de candidats détectés : {len(colonnes_voix)}")

# # Convertir TOUTES les colonnes de voix en numérique
# for col in colonnes_voix:
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# # Fonction pour identifier le gagnant parmi TOUS les candidats
# def identifier_gagnant(row):
#     voix_values = {}
    
#     # Parcourir TOUTES les colonnes de voix
#     for col in colonnes_voix:
#         # Extraire le numéro du candidat depuis le nom de la colonne "Voix X"
#         numero_candidat = col.split(" ")[-1]
#         voix_value = row[col]
        
#         # Ajouter seulement si c'est un nombre valide
#         if pd.notna(voix_value) and voix_value >= 0:
#             voix_values[numero_candidat] = voix_value
    
#     # Debug : afficher les voix pour la première ligne
#     if row.name == 0:
#         print(f"Voix pour la première ligne : {voix_values}")
    
#     # Trouver le candidat avec le MAXIMUM de voix parmi TOUS
#     if voix_values:
#         gagnant = max(voix_values, key=voix_values.get)
#         return gagnant
#     return None

# # Appliquer l'identification du gagnant
# df["Index Gagnant"] = df.apply(identifier_gagnant, axis=1)

# # Fonction pour récupérer la valeur correspondant au gagnant
# def get_value_by_index(row, base_column):
#     if pd.isna(row['Index Gagnant']):
#         return None
#     column_name = f"{base_column} {row['Index Gagnant']}"
#     return row.get(column_name, None)

# # Créer les colonnes pour l'élu (maintenant ça marche pour TOUS les candidats)
# df["Sexe Élu"] = df.apply(lambda row: get_value_by_index(row, "Sexe"), axis=1)
# df["Nom Élu"] = df.apply(lambda row: get_value_by_index(row, "Nom"), axis=1)
# df["Prénom Élu"] = df.apply(lambda row: get_value_by_index(row, "Prénom"), axis=1)
# df["Voix Élu"] = df.apply(lambda row: get_value_by_index(row, "Voix"), axis=1)
# df["% Voix/Ins Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Ins"), axis=1)
# df["% Voix/Exp Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Exp"), axis=1)

# # Vérification pour debug
# print(f"Nombre de gagnants identifiés : {df['Index Gagnant'].notna().sum()}")
# print(f"Index gagnants uniques : {df['Index Gagnant'].unique()}")

# # === 8. Ajouter colonne Parti Politique Élu ===
# dictionnaire_partis = {
#     "macron": "Renaissance", "philippe": "Horizons", "bayrou": "MoDem",
#     "mélenchon": "LFI", "autain": "LFI", "corbière": "LFI", "panot": "LFI",
#     "le pen": "RN", "bardella": "RN", "mariani": "RN",
#     "zemmour": "Reconquête", "marion maréchal": "Reconquête",
#     "pecresse": "LR", "ciotti": "LR", "jacob": "LR", "fillon": "LR",
#     "hidalgo": "PS", "faure": "PS", "rousset": "PS",
#     "jadot": "EELV", "rousseau": "EELV", "garrido": "EELV",
#     "roussel": "PCF", "lautrette": "PCF",
#     "asselineau": "UPR", "hamon": "Génération.s",
#     "lassalle": "Résistons", "arthaud": "LO", "poutou": "NPA",
#     "dupont-aignan": "DLF","pécresse":"LR","Valérie":"LR"
# }

# def associer_parti(nom):
#     if pd.isna(nom):
#         return None
#     nom = nom.lower()
#     for cle in dictionnaire_partis:
#         if cle in nom:
#             return dictionnaire_partis[cle]
#     return "Inconnu"

# df["Parti Politique Élu"] = df["Nom Élu"].apply(associer_parti)

# # === 9. Orientation politique & score ===
# orientation_politique = {
#     "LFI": ("Extrême gauche", -5), "PCF": ("Gauche", -4), "Génération.s": ("Gauche", -4),
#     "EELV": ("Gauche", -3), "PS": ("Centre gauche", -2),
#     "Renaissance": ("Centre", 0), "MoDem": ("Centre", 0), "Horizons": ("Centre", 1),
#     "LR": ("Droite", 3), "DLF": ("Droite", 4),
#     "RN": ("Extrême droite", 5), "Reconquête": ("Extrême droite", 5),
#     "UPR": ("Souverainiste", 3), "Résistons": ("Divers", 0),
#     "LO": ("Extrême gauche", -5), "NPA": ("Extrême gauche", -5)
# }

# def get_orientation(parti):
#     if pd.isna(parti) or parti == "Inconnu":
#         return ("Inconnu", None)
#     return orientation_politique.get(parti, ("Divers", 0))

# df[["Orientation Politique", "Score Orientation (-5 à 5)"]] = df["Parti Politique Élu"].apply(
#     lambda p: pd.Series(get_orientation(p))
# )

# # === Vérification ===
# print("\n Analyse de la qualité des colonnes principales :")
# print("--------------------------------------------------")
# colonnes_cles = ["ID Commune","Nom Élu", "Prénom Élu", "Voix Élu", "Parti Politique Élu"]
# for col in colonnes_cles:
#     nb_nan = df[col].isna().sum()
#     nb_inconnu = (df[col] == "Inconnu").sum() if df[col].dtype == object else 0
#     print(f"{col} → NaN: {nb_nan}, Inconnu: {nb_inconnu}")
# print("--------------------------------------------------")

# # === Nettoyage final ===
# Code_de_la_commune = [col for col in df.columns if col.startswith("Code de la commune")]
# Code_du_département = [col for col in df.columns if col.startswith("Code du département")]
# N_Panneau = [col for col in df.columns if col.startswith("N°Panneau")]

# colonnes_a_supprimer = ["Index Gagnant"] + N_Panneau + Code_du_département + Code_de_la_commune + colonnes_voix + colonnes_nom + colonnes_prenom + colonnes_sexe + colonnes_panneau + colonnes_pourcent_ins + colonnes_pourcent_exp
# colonnes_a_conserver = ["ID Commune", "Abstentions", "Inscrits"]
# df = df[[col for col in df.columns if col not in colonnes_a_supprimer or col in colonnes_a_conserver]]

# # === Réorganiser ===
# colonnes_ordre = [
#     "ID Commune", "Inscrits", "Abstentions",
#     "Sexe Élu", "Nom Élu", "Prénom Élu", "Parti Politique Élu",
#     "Orientation Politique", "Score Orientation (-5 à 5)",
#     "Voix Élu", "% Voix/Ins Élu", "% Voix/Exp Élu"
# ]
# df = df[[col for col in colonnes_ordre if col in df.columns]]

# # === Export ===
# def sauvegarder_fichier_csv(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer"):
#     os.makedirs(dossier, exist_ok=True)
#     chemin_complet = os.path.join(dossier, nom_fichier)
#     try:
#         df.to_csv(chemin_complet, index=False, encoding='utf-8-sig', sep=';')
#         print(f" Fichier CSV enregistré dans : {chemin_complet}")
#     except Exception as e:
#         print(" Erreur lors de l'enregistrement CSV :", e)
# def sauvegarder_fichier(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer"):
#     os.makedirs(dossier, exist_ok=True)
#     chemin_complet = os.path.join(dossier, nom_fichier)
#     try:
#         df.to_excel(chemin_complet, index=False, engine='openpyxl')
#         print(f" Fichier nettoyé enregistré dans : {chemin_complet}")
#     except Exception as e:
#         print(" Erreur lors de l'enregistrement :", e)

# df_nom_fichier = "DATA_Nettoyer_t1_2022.xlsx"
# sauvegarder_fichier(df, df_nom_fichier)

# # Sauvegarder en CSV
# df_nom_fichier_csv = "DATA_Nettoyer_t1_2022.csv"
# sauvegarder_fichier_csv(df, df_nom_fichier_csv)
import os
import pandas as pd
from collections import Counter

# === 1. Charger le fichier Excel ===
fichier_source = "C:/Users/Massi/Desktop/MSPR BLOC 03/DATA_Set_brut/2022_tour1.xlsx"
feuille = "Commune"
df_brut = pd.read_excel(fichier_source, sheet_name=feuille)

# === 2. Détecter automatiquement la ligne d'en-tête ===
ligne_entete_index = df_brut[df_brut.eq("Code du département").any(axis=1)].index[0]
df = df_brut[ligne_entete_index:].reset_index(drop=True)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# === 3. Nettoyage des colonnes vides et caractères spéciaux ===
# NE PAS supprimer les colonnes sans nom pour l'instant, on va les renommer
df.columns = df.columns.astype(str)  # Convertir en string pour éviter les erreurs
df.columns = df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True)
df.columns = df.columns.str.strip()

print("Colonnes avant renommage :")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")
print("--------------------------------------------------")

# === 4. Vérifier les colonnes dupliquées ===
compte_colonnes = Counter(df.columns)
colonnes_dupliquees = {col: count for col, count in compte_colonnes.items() if count > 1}
nb_lignes_dupliquees = df.duplicated().sum()

print(" Analyse du fichier COMMUNE :")
print("--------------------------------------------------")
print(f"Nombre total de colonnes         : {len(df.columns)}")
print(f"Nombre de colonnes uniques       : {len(set(df.columns))}")
print(f"Nombre de colonnes dupliquées    : {len(colonnes_dupliquees)}")
print(f"Nombre de lignes dupliquées      : {nb_lignes_dupliquees}")
print("--------------------------------------------------")

# === 5. Renommer dynamiquement TOUTES les colonnes candidats ===
def renommer_colonnes_candidats(colonnes: list) -> list:
    nouvelles_colonnes = []
    compteur_candidat = 0
    sequence_candidat = ["N°Panneau", "Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]
    position_dans_sequence = 0
    candidat_commence = False
    
    for i, col in enumerate(colonnes):
        # Détecter le début des données candidats
        if col == "N°Panneau" and not candidat_commence:
            candidat_commence = True
            compteur_candidat = 1
            position_dans_sequence = 0
            nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
            position_dans_sequence += 1
        elif candidat_commence:
            # Si on est dans une séquence de candidat
            if col in sequence_candidat or col == "nan" or col.strip() == "" or pd.isna(col):
                # Si c'est le début d'un nouveau candidat (N°Panneau)
                if position_dans_sequence == 0 or position_dans_sequence >= len(sequence_candidat):
                    compteur_candidat += 1
                    position_dans_sequence = 0
                
                # Assigner le nom de la colonne selon la position dans la séquence
                if position_dans_sequence < len(sequence_candidat):
                    nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
                    position_dans_sequence += 1
                else:
                    # Si on dépasse la séquence, on recommence
                    compteur_candidat += 1
                    position_dans_sequence = 0
                    nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
                    position_dans_sequence += 1
            else:
                # Si on rencontre une colonne qui n'est pas dans la séquence candidat
                # on continue la séquence actuelle
                if position_dans_sequence < len(sequence_candidat):
                    nouvelles_colonnes.append(f"{sequence_candidat[position_dans_sequence]} {compteur_candidat}")
                    position_dans_sequence += 1
                else:
                    nouvelles_colonnes.append(col)
        else:
            # Colonnes avant les candidats
            nouvelles_colonnes.append(col)
    
    return nouvelles_colonnes

def appliquer_renommage(df):
    df.columns = renommer_colonnes_candidats(list(df.columns))
    return df

# APPLIQUER LE RENOMMAGE D'ABORD
df = appliquer_renommage(df)

print("Colonnes après renommage :")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")
print("--------------------------------------------------")

# === 6. Créer la colonne ID Commune ===
df["Code du département"] = df["Code du département"].astype(str).str.zfill(2)
df["Code de la commune"] = df["Code de la commune"].astype(str).str.zfill(3)
df["ID Commune"] = df["Code du département"] + df["Code de la commune"]
# === AJOUT DE LA COLONNE ANNÉE ===
# Pour le dataset 2022 - changez cette valeur selon le dataset traité
df["Année"] = 2022  # Changez en 2017 pour l'autre dataset
# === 7. Identifier l'élu APRÈS le renommage ===
# Maintenant on récupère TOUTES les colonnes renommées
colonnes_voix = [col for col in df.columns if col.startswith("Voix ")]
colonnes_nom = [col for col in df.columns if col.startswith("Nom ")]
colonnes_prenom = [col for col in df.columns if col.startswith("Prénom ")]
colonnes_sexe = [col for col in df.columns if col.startswith("Sexe ")]
colonnes_panneau = [col for col in df.columns if col.startswith("N°Panneau ")]
colonnes_pourcent_ins = [col for col in df.columns if col.startswith("% Voix/Ins ")]
colonnes_pourcent_exp = [col for col in df.columns if col.startswith("% Voix/Exp ")]

print(f"Colonnes de voix trouvées : {colonnes_voix}")  # Debug pour vérifier
print(f"Colonnes de nom trouvées : {colonnes_nom}")
print(f"Nombre de candidats détectés : {len(colonnes_voix)}")

# Convertir TOUTES les colonnes de voix en numérique
for col in colonnes_voix:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Fonction pour identifier le gagnant parmi TOUS les candidats
def identifier_gagnant(row):
    voix_values = {}
    
    # Parcourir TOUTES les colonnes de voix
    for col in colonnes_voix:
        # Extraire le numéro du candidat depuis le nom de la colonne "Voix X"
        numero_candidat = col.split(" ")[-1]
        voix_value = row[col]
        
        # Ajouter seulement si c'est un nombre valide
        if pd.notna(voix_value) and voix_value >= 0:
            voix_values[numero_candidat] = voix_value
    
    # Debug : afficher les voix pour la première ligne
    if row.name == 0:
        print(f"Voix pour la première ligne : {voix_values}")
    
    # Trouver le candidat avec le MAXIMUM de voix parmi TOUS
    if voix_values:
        gagnant = max(voix_values, key=voix_values.get)
        return gagnant
    return None

# Appliquer l'identification du gagnant
df["Index Gagnant"] = df.apply(identifier_gagnant, axis=1)

# Fonction pour récupérer la valeur correspondant au gagnant
def get_value_by_index(row, base_column):
    if pd.isna(row['Index Gagnant']):
        return None
    column_name = f"{base_column} {row['Index Gagnant']}"
    return row.get(column_name, None)

# Créer les colonnes pour l'élu (maintenant ça marche pour TOUS les candidats)
df["Sexe Élu"] = df.apply(lambda row: get_value_by_index(row, "Sexe"), axis=1)
df["Nom Élu"] = df.apply(lambda row: get_value_by_index(row, "Nom"), axis=1)
df["Prénom Élu"] = df.apply(lambda row: get_value_by_index(row, "Prénom"), axis=1)
df["Voix Élu"] = df.apply(lambda row: get_value_by_index(row, "Voix"), axis=1)
df["% Voix/Ins Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Ins"), axis=1)
df["% Voix/Exp Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Exp"), axis=1)

# Concaténer le nom et prénom de l'élu
def concatener_nom_prenom(row):
    nom = row["Nom Élu"] if pd.notna(row["Nom Élu"]) else ""
    prenom = row["Prénom Élu"] if pd.notna(row["Prénom Élu"]) else ""
    
    if nom and prenom:
        return f"{nom} {prenom}"
    elif nom:
        return nom
    elif prenom:
        return prenom
    else:
        return None

df["Nom Complet Élu"] = df.apply(concatener_nom_prenom, axis=1)

# Vérification pour debug
print(f"Nombre de gagnants identifiés : {df['Index Gagnant'].notna().sum()}")
print(f"Index gagnants uniques : {df['Index Gagnant'].unique()}")

# === 8. Ajouter colonne Parti Politique Élu ===
dictionnaire_partis = {
    "macron": "Renaissance", "philippe": "Horizons", "bayrou": "MoDem",
    "mélenchon": "LFI", "autain": "LFI", "corbière": "LFI", "panot": "LFI",
    "le pen": "RN", "bardella": "RN", "mariani": "RN",
    "zemmour": "Reconquête", "marion maréchal": "Reconquête",
    "pecresse": "LR", "ciotti": "LR", "jacob": "LR", "fillon": "LR",
    "hidalgo": "PS", "faure": "PS", "rousset": "PS",
    "jadot": "EELV", "rousseau": "EELV", "garrido": "EELV",
    "roussel": "PCF", "lautrette": "PCF",
    "asselineau": "UPR", "hamon": "Génération.s",
    "lassalle": "Résistons", "arthaud": "LO", "poutou": "NPA",
    "dupont-aignan": "DLF","pécresse":"LR","Valérie":"LR"
}

def associer_parti(nom):
    if pd.isna(nom):
        return None
    nom = nom.lower()
    for cle in dictionnaire_partis:
        if cle in nom:
            return dictionnaire_partis[cle]
    return "Inconnu"

df["Parti Politique Élu"] = df["Nom Élu"].apply(associer_parti)

# === 9. Orientation politique & score (NOUVELLE VERSION) ===
orientation_politique = {
    "LFI": "Extrême gauche", "PCF": "Gauche", "Génération.s": "Gauche",
    "EELV": "Gauche", "PS": "Gauche",
    "Renaissance": "Centre", "MoDem": "Centre", "Horizons": "Centre",
    "LR": "Droite", "DLF": "Droite", "UPR": "Droite",
    "RN": "Extrême droite", "Reconquête": "Extrême droite",
    "Résistons": "Divers", "LO": "Extrême gauche", "NPA": "Extrême gauche"
}

# Affectation de l'orientation politique par parti
df["Orientation Politique"] = df["Parti Politique Élu"].apply(
    lambda p: orientation_politique.get(p, "Inconnu") if pd.notna(p) else "Inconnu"
)

# Nouvelle échelle de score : selon l’orientation (et plus le parti)
echelle_orientation = {
    "Extrême gauche": 0,
    "Gauche": 1,
    "Centre": 2,
    "Droite": 3,
    "Extrême droite": 4
}

# Attribution du score
df["Score Orientation (0 à 4)"] = df["Orientation Politique"].apply(
    lambda orientation: echelle_orientation.get(orientation, None)
)


# === Vérification ===
print("\n Analyse de la qualité des colonnes principales :")
print("--------------------------------------------------")
colonnes_cles = ["ID Commune","Nom Complet Élu", "Voix Élu", "Parti Politique Élu"]
for col in colonnes_cles:
    nb_nan = df[col].isna().sum()
    nb_inconnu = (df[col] == "Inconnu").sum() if df[col].dtype == object else 0
    print(f"{col} → NaN: {nb_nan}, Inconnu: {nb_inconnu}")
print("--------------------------------------------------")

# === Nettoyage final ===
Code_de_la_commune = [col for col in df.columns if col.startswith("Code de la commune")]
Code_du_département = [col for col in df.columns if col.startswith("Code du département")]
N_Panneau = [col for col in df.columns if col.startswith("N°Panneau")]

colonnes_a_supprimer = ["Index Gagnant"] + N_Panneau + Code_du_département + Code_de_la_commune + colonnes_voix + colonnes_nom + colonnes_prenom + colonnes_sexe + colonnes_panneau + colonnes_pourcent_ins + colonnes_pourcent_exp
colonnes_a_conserver = ["ID Commune", "Abstentions", "Inscrits"]
df = df[[col for col in df.columns if col not in colonnes_a_supprimer or col in colonnes_a_conserver]]

# === Réorganiser ===
colonnes_ordre = [
    "ID Commune", "Année","Inscrits", "Abstentions",
    "Sexe Élu", "Nom Complet Élu", "Parti Politique Élu",
    "Orientation Politique",  "Score Orientation (0 à 4)",
    "Voix Élu", "% Voix/Ins Élu", "% Voix/Exp Élu"
]
df = df[[col for col in colonnes_ordre if col in df.columns]]

# === Export ===
def sauvegarder_fichier_excel(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_elections"):
    os.makedirs(dossier, exist_ok=True)
    chemin_complet = os.path.join(dossier, nom_fichier)
    try:
        df.to_excel(chemin_complet, index=False, engine='openpyxl')
        print(f" Fichier Excel enregistré dans : {chemin_complet}")
    except Exception as e:
        print(" Erreur lors de l'enregistrement Excel :", e)

def sauvegarder_fichier_csv(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer/DATA_elections"):
    os.makedirs(dossier, exist_ok=True)
    chemin_complet = os.path.join(dossier, nom_fichier)
    try:
        df.to_csv(chemin_complet, index=False, encoding='utf-8-sig', sep=';')
        print(f" Fichier CSV enregistré dans : {chemin_complet}")
    except Exception as e:
        print(" Erreur lors de l'enregistrement CSV :", e)

# Sauvegarder en Excel
df_nom_fichier_excel = "DATA_Nettoyer_t1_2022.xlsx"
sauvegarder_fichier_excel(df, df_nom_fichier_excel)

# Sauvegarder en CSV
df_nom_fichier_csv = "DATA_Nettoyer_t1_2022.csv"
sauvegarder_fichier_csv(df, df_nom_fichier_csv)