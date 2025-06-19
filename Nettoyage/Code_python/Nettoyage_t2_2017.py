# import os
# import pandas as pd
# from collections import Counter

# # === 1. Charger le fichier Excel ===
# fichier_source = "C:/Users/Massi/Desktop/MSPR BLOC 03/DATA_Set_brut/2017_tour2.xlsx"
# feuille = "commune"
# df_brut = pd.read_excel(fichier_source, sheet_name=feuille)

# # === 2. Détecter automatiquement la ligne d'en-tête
# ligne_entete_index = df_brut[df_brut.eq("Code du département").any(axis=1)].index[0]
# df = df_brut[ligne_entete_index:].reset_index(drop=True)
# df.columns = df.iloc[0]
# df = df[1:].reset_index(drop=True)

# # === 3. Nettoyage des colonnes vides et caractères spéciaux
# df = df.loc[:, ~df.columns.isnull()]  # Supprime colonnes sans nom
# df.columns = df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True)  # Nettoie les noms
# df.columns = df.columns.str.strip()  # Trim des colonnes

# # === 4. Vérifier les colonnes dupliquées (juste affichage, pas de suppression)
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

# # === 5. Renommer dynamiquement les colonnes candidats (en blocs commençant par Sexe)
# def renommer_colonnes_candidats(colonnes: list) -> list:
#     nouvelles_colonnes = []
#     compteur_candidat = 0
#     after_static_block = False
#     for col in colonnes:
#         if not after_static_block and col == "Sexe":
#             after_static_block = True
#         if after_static_block and col in ["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp", "N°Panneau"]:
#             if col == "Sexe":
#                 compteur_candidat += 1
#             nouvelles_colonnes.append(f"{col} {compteur_candidat}")
#         else:
#             nouvelles_colonnes.append(col)
#     return nouvelles_colonnes

# def appliquer_renommage(df):
#     df.columns = renommer_colonnes_candidats(list(df.columns))
#     return df

# df = appliquer_renommage(df)

# # === 6. Créer la colonne ID Commune (on garde les colonnes d'origine)
# df["Code du département"] = df["Code du département"].astype(str).str.zfill(2)
# df["Code de la commune"] = df["Code de la commune"].astype(str).str.zfill(3)
# df["ID Commune"] = df["Code du département"] +  df["Code de la commune"]

# # === 7. Identifier l'élu (candidat ayant obtenu le plus de voix par commune)
# colonnes_voix = [col for col in df.columns if col.startswith("Voix ")]
# colonnes_nom = [col for col in df.columns if col.startswith("Nom ")]
# colonnes_prenom = [col for col in df.columns if col.startswith("Prénom ")]
# colonnes_sexe = [col for col in df.columns if col.startswith("Sexe ")]
# colonnes_panneau = [col for col in df.columns if col.startswith("N°Panneau ")]
# colonnes_pourcent_ins = [col for col in df.columns if col.startswith("% Voix/Ins ")]
# colonnes_pourcent_exp = [col for col in df.columns if col.startswith("% Voix/Exp ")]
# Code_de_la_commune = [col for col in df.columns if col.startswith("Code de la commune ")]
# Code_du_département = [col for col in df.columns if col.startswith("Code du département")]
# N_Panneau = [col for col in df.columns if col.startswith("N°Panneau")]

# # Pour chaque ligne, trouver l'index du max parmi les colonnes de voix
# df["Index Gagnant"] = df[colonnes_voix].astype(float).idxmax(axis=1).str.extract(r'(\d+)$').astype(int)

# # Utiliser cet index pour retrouver toutes les infos du gagnant
# get_value_by_index = lambda row, base: row.get(f"{base} {row['Index Gagnant']}", None)

# df["Sexe Élu"] = df.apply(lambda row: get_value_by_index(row, "Sexe"), axis=1)
# df["Nom Élu"] = df.apply(lambda row: get_value_by_index(row, "Nom"), axis=1)
# df["Prénom Élu"] = df.apply(lambda row: get_value_by_index(row, "Prénom"), axis=1)
# df["Voix Élu"] = df.apply(lambda row: get_value_by_index(row, "Voix"), axis=1)
# df["% Voix/Ins Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Ins"), axis=1)
# df["% Voix/Exp Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Exp"), axis=1)

# # === 8. Ajouter colonne Parti Politique Élu (matching étendu)
# dictionnaire_partis = {
#     "macron": "Renaissance", "philippe": "Horizons", "bayrou": "MoDem",
#     "mélenchon": "LFI", "autain": "LFI", "corbière": "LFI", "panot": "LFI",
#     "le pen": "RN", "bardella": "RN", "mariani": "RN",
#     "zemmour": "Reconquête", "marion maréchal": "Reconquête",
#     "pecresse": "LR", "ciotti": "LR", "jacob": "LR",
#     "hidalgo": "PS", "faure": "PS", "rousset": "PS",
#     "jadot": "EELV", "rousseau": "EELV", "garrido": "EELV",
#     "roussel": "PCF", "lautrette": "PCF",
#     "fillon": "LR","asselineau": "UPR", "hamon": "Génération.s",
#     "lassalle": "Résistons", "arthaud": "LO", "poutou": "NPA",
#     "dupont-aignan": "DLF"
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


# # === Vérification des valeurs manquantes ou inconnues
# print("\n Analyse de la qualité des colonnes principales :")
# print("--------------------------------------------------")
# colonnes_cles = ["ID Commune","Nom Élu", "Prénom Élu", "Voix Élu", "Parti Politique Élu"]
# for col in colonnes_cles:
#     nb_nan = df[col].isna().sum()
#     nb_inconnu = (df[col] == "Inconnu").sum() if df[col].dtype == object else 0
#     print(f"{col} → NaN: {nb_nan}, Inconnu: {nb_inconnu}")
# print("--------------------------------------------------")


# # Supprimer uniquement les colonnes des autres candidats
# colonnes_a_supprimer = ["Index Gagnant"] + N_Panneau + Code_du_département + Code_de_la_commune  + colonnes_voix + colonnes_nom + colonnes_prenom + colonnes_sexe + colonnes_panneau + colonnes_pourcent_ins + colonnes_pourcent_exp
# colonnes_a_conserver = ["ID Commune", "Abstentions", "Inscrits"]
# df = df[[col for col in df.columns if col not in colonnes_a_supprimer or col in colonnes_a_conserver]]

# # === Réorganiser les colonnes
# colonnes_ordre = [
#     "ID Commune", "Inscrits", "Abstentions",
#     "Sexe Élu", "Nom Élu", "Prénom Élu", "Parti Politique Élu",
#      "Voix Élu", "% Voix/Ins Élu", "% Voix/Exp Élu"
# ]
# df = df[[col for col in colonnes_ordre if col in df.columns]]



# # === 8. Export du fichier Excel sécurisé
# def sauvegarder_fichier(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer"):
#     os.makedirs(dossier, exist_ok=True)
#     chemin_complet = os.path.join(dossier, nom_fichier)
#     try:
#         df.to_excel(chemin_complet, index=False, engine='openpyxl')
#         print(f" Fichier nettoyé enregistré dans : {chemin_complet}")
#     except Exception as e:
#         print(" Erreur lors de l'enregistrement :", e)

# # Sauvegarde
# df_nom_fichier = "DATA_Nettoyer_t2.xlsx"
# sauvegarder_fichier(df, df_nom_fichier)
import os
import pandas as pd
from collections import Counter

# # === 1. Charger le fichier Excel ===
fichier_source = "C:/Users/Massi/Desktop/MSPR BLOC 03/DATA_Set_brut/2017_tour2.xlsx"
feuille = "commune"
df_brut = pd.read_excel(fichier_source, sheet_name=feuille)

# === 2. Détecter automatiquement la ligne d'en-tête
ligne_entete_index = df_brut[df_brut.eq("Code du département").any(axis=1)].index[0]
df = df_brut[ligne_entete_index:].reset_index(drop=True)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# === 3. Nettoyage des colonnes vides et caractères spéciaux
# df = df.loc[:, ~df.columns.isnull()]  # Supprime colonnes sans nom
df.columns = df.columns.str.replace(r'[\n\r\t]+', ' ', regex=True)  # Nettoie les noms
df.columns = df.columns.str.strip()  # Trim des colonnes

# === 4. Vérifier les colonnes dupliquées (juste affichage, pas de suppression)
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

# === 5. Renommer dynamiquement les colonnes candidats (en blocs commençant par Sexe)
def renommer_colonnes_candidats(colonnes: list) -> list:
    nouvelles_colonnes = []
    compteur_candidat = 0
    after_static_block = False
    for col in colonnes:
        if not after_static_block and col == "Sexe":
            after_static_block = True
        if after_static_block and col in ["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp", "N°Panneau"]:
            if col == "Sexe":
                compteur_candidat += 1
            nouvelles_colonnes.append(f"{col} {compteur_candidat}")
        else:
            nouvelles_colonnes.append(col)
    return nouvelles_colonnes

def appliquer_renommage(df):
    df.columns = renommer_colonnes_candidats(list(df.columns))
    return df

df = appliquer_renommage(df)

# === 6. Créer la colonne ID Commune (on garde les colonnes d'origine)
df["Code du département"] = df["Code du département"].astype(str).str.zfill(2)
df["Code de la commune"] = df["Code de la commune"].astype(str).str.zfill(3)
df["ID Commune"] = df["Code du département"] +  df["Code de la commune"]

# === 7. Identifier l'élu (candidat ayant obtenu le plus de voix par commune)
colonnes_voix = [col for col in df.columns if col.startswith("Voix ")]
colonnes_nom = [col for col in df.columns if col.startswith("Nom ")]
colonnes_prenom = [col for col in df.columns if col.startswith("Prénom ")]
colonnes_sexe = [col for col in df.columns if col.startswith("Sexe ")]
colonnes_panneau = [col for col in df.columns if col.startswith("N°Panneau ")]
colonnes_pourcent_ins = [col for col in df.columns if col.startswith("% Voix/Ins ")]
colonnes_pourcent_exp = [col for col in df.columns if col.startswith("% Voix/Exp ")]

# Pour chaque ligne, trouver l'index du max parmi les colonnes de voix
df["Index Gagnant"] = df[colonnes_voix].astype(float).idxmax(axis=1).str.extract(r'(\d+)$').astype(int)

# Utiliser cet index pour retrouver toutes les infos du gagnant
get_value_by_index = lambda row, base: row.get(f"{base} {row['Index Gagnant']}", None)

df["Sexe Élu"] = df.apply(lambda row: get_value_by_index(row, "Sexe"), axis=1)
df["Nom Élu"] = df.apply(lambda row: get_value_by_index(row, "Nom"), axis=1)
df["Prénom Élu"] = df.apply(lambda row: get_value_by_index(row, "Prénom"), axis=1)
df["Voix Élu"] = df.apply(lambda row: get_value_by_index(row, "Voix"), axis=1)
df["Nom Complet Élu"] = df["Prénom Élu"].fillna('') + ' ' + df["Nom Élu"].fillna('')
df["Nom Complet Élu"] = df["Nom Complet Élu"].str.strip()
df["% Voix/Ins Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Ins"), axis=1)
df["% Voix/Exp Élu"] = df.apply(lambda row: get_value_by_index(row, "% Voix/Exp"), axis=1)

# === 8. Ajouter colonne Parti Politique Élu (matching étendu)
dictionnaire_partis = {
    "macron": "Renaissance", "philippe": "Horizons", "bayrou": "MoDem",
    "mélenchon": "LFI", "autain": "LFI", "corbière": "LFI", "panot": "LFI",
    "le pen": "RN", "bardella": "RN", "mariani": "RN",
    "zemmour": "Reconquête", "marion maréchal": "Reconquête",
    "pecresse": "LR", "ciotti": "LR", "jacob": "LR",
    "hidalgo": "PS", "faure": "PS", "rousset": "PS",
    "jadot": "EELV", "rousseau": "EELV", "garrido": "EELV",
    "roussel": "PCF", "lautrette": "PCF",
    "fillon": "LR","asselineau": "UPR", "hamon": "Génération.s",
    "lassalle": "Résistons", "arthaud": "LO", "poutou": "NPA",
    "dupont-aignan": "DLF"
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

# === 9. Ajouter colonne Orientation Politique (-5 à 5) selon le parti
def orientation_politique(parti):
    if pd.isna(parti):
        return None
    mapping_orientation = {
    "LFI": -5,
    "PCF": -5,
    "LO": -5,
    "NPA": -5,
    "Génération.s": -3,
    "PS": -2,
    "EELV": -1,
    "MoDem": 0,
    "Renaissance": 0,
    "Horizons": 2,
    "Résistons": 3,
    "LR": 4,
    "DLF": 4,
    "UPR": 4,
    "RN": 5,
    "Reconquête": 5

    }
    return mapping_orientation.get(parti, None)

df["Orientation Politique"] = df["Parti Politique Élu"].apply(orientation_politique)

# === 10. Calcul du taux d'abstention (%)
df["Abstentions"] = pd.to_numeric(df["Abstentions"], errors='coerce')
df["Inscrits"] = pd.to_numeric(df["Inscrits"], errors='coerce')
df["Taux Abstention (%)"] = (df["Abstentions"] / df["Inscrits"]) * 100

# === Vérification des valeurs manquantes ou inconnues
print("\n Analyse de la qualité des colonnes principales :")
print("--------------------------------------------------")
colonnes_cles = ["ID Commune", "Nom Élu", "Prénom Élu", "Voix Élu", "Parti Politique Élu", "Orientation Politique", "Taux Abstention (%)"]
for col in colonnes_cles:
    nb_nan = df[col].isna().sum()
    nb_inconnu = (df[col] == "Inconnu").sum() if df[col].dtype == object else 0
    print(f"{col} → NaN: {nb_nan}, Inconnu: {nb_inconnu}")
print("--------------------------------------------------")

# Supprimer uniquement les colonnes des autres candidats
colonnes_panneau = [col for col in df.columns if col.startswith("N°Panneau")]
colonnes_code_dep = [col for col in df.columns if col.startswith("Code du département")]
colonnes_code_com = [col for col in df.columns if col.startswith("Code de la commune")]

colonnes_a_supprimer = (
    ["Index Gagnant"] + colonnes_panneau + colonnes_code_dep + colonnes_code_com +
    colonnes_voix + colonnes_nom + colonnes_prenom + colonnes_sexe +
    colonnes_pourcent_ins + colonnes_pourcent_exp
)

colonnes_a_conserver = [
    "ID Commune", "Abstentions", "Inscrits", "Taux Abstention (%)",
    "Sexe Élu", "Nom Complet Élu", "Parti Politique Élu", "Orientation Politique",
   "% Voix/Exp Élu"
]

df = df[[col for col in df.columns if col not in colonnes_a_supprimer or col in colonnes_a_conserver]]

# === Réorganiser les colonnes
df = df[[col for col in colonnes_a_conserver if col in df.columns]]

# === 11. Export du fichier Excel sécurisé
def sauvegarder_fichier(df, nom_fichier, dossier="C:/Users/Massi/Desktop/MSPR BLOC 03/Nettoyage/DATA_Nettoyer"):
    os.makedirs(dossier, exist_ok=True)
    chemin_complet = os.path.join(dossier, nom_fichier)
    try:
        df.to_excel(chemin_complet, index=False, engine='openpyxl')
        print(f" Fichier nettoyé enregistré dans : {chemin_complet}")
    except Exception as e:
        print(" Erreur lors de l'enregistrement :", e)

# Sauvegarde
df_nom_fichier = "DATA_Nettoyer_t2_2017.xlsx"
sauvegarder_fichier(df, df_nom_fichier)
