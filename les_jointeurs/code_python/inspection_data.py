import pandas as pd
import os

# === Chemins (ajustez selon vos chemins réels) ===
fichier_elections = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Nettoyer_Elections.csv"
fichier_population = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Population_finale.csv"
fichier_criminaliter = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Criminalite_2021_2016.csv"
fichier_emploi = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/DATA_Emploi_2022_2017.csv"

def inspecter_dataset(fichier_path, nom_dataset):
    """Inspecte un dataset et affiche ses informations"""
    print(f"\n{'='*60}")
    print(f"📊 DATASET: {nom_dataset}")
    print(f"📁 Fichier: {fichier_path}")
    print(f"{'='*60}")
    
    try:
        # Essayer différents séparateurs
        separateurs = [';', ',', '\t']
        df = None
        
        for sep in separateurs:
            try:
                df_test = pd.read_csv(fichier_path, sep=sep, nrows=5)
                if len(df_test.columns) > 1:  # Si plus d'une colonne, c'est probablement le bon séparateur
                    df = pd.read_csv(fichier_path, sep=sep)
                    print(f"✅ Séparateur détecté: '{sep}'")
                    break
            except Exception as e:
                continue
        
        if df is None:
            print("❌ Impossible de lire le fichier avec les séparateurs testés")
            return
        
        print(f"📏 Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        print(f"\n📋 COLONNES DISPONIBLES:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. '{col}'")
        
        print(f"\n🔍 ÉCHANTILLON DES DONNÉES (5 premières lignes):")
        print(df.head().to_string())
        
        print(f"\n📊 TYPES DE DONNÉES:")
        for col in df.columns:
            print(f"   - {col}: {df[col].dtype}")
        
        print(f"\n📈 VALEURS MANQUANTES:")
        valeurs_manquantes = df.isnull().sum()
        for col in df.columns:
            if valeurs_manquantes[col] > 0:
                print(f"   - {col}: {valeurs_manquantes[col]} valeurs manquantes")
        
        if valeurs_manquantes.sum() == 0:
            print("   ✅ Aucune valeur manquante détectée")
        
        # Vérifier les colonnes d'ID et d'année
        print(f"\n🔑 COLONNES CLÉS DÉTECTÉES:")
        colonnes_possibles_id = [col for col in df.columns if 'commune' in col.lower() or 'id' in col.lower()]
        colonnes_possibles_annee = [col for col in df.columns if 'annee' in col.lower() or 'année' in col.lower() or 'year' in col.lower()]
        
        if colonnes_possibles_id:
            print(f"   ID Commune possible: {colonnes_possibles_id}")
        if colonnes_possibles_annee:
            print(f"   Année possible: {colonnes_possibles_annee}")
        
        # Vérifier les valeurs uniques pour les colonnes clés
        if colonnes_possibles_annee:
            for col in colonnes_possibles_annee:
                annees_uniques = sorted(df[col].unique())
                print(f"   Années dans '{col}': {annees_uniques}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier: {str(e)}")

# === Inspection de tous les datasets ===
print("🔍 INSPECTION DES DATASETS")
print("="*80)

datasets = [
    (fichier_elections, "ELECTIONS"),
    (fichier_population, "POPULATION"),
    (fichier_criminaliter, "CRIMINALITÉ"),
    (fichier_emploi, "EMPLOI")
]

for fichier, nom in datasets:
    if os.path.exists(fichier):
        inspecter_dataset(fichier, nom)
    else:
        print(f"\n❌ FICHIER NON TROUVÉ: {nom}")
        print(f"   Chemin: {fichier}")

print(f"\n{'='*80}")
print("🎯 RÉSUMÉ DES ACTIONS À EFFECTUER:")
print("="*80)
print("1. Vérifiez les noms exacts des colonnes dans chaque dataset")
print("2. Ajustez les listes 'colonnes_elec', 'colonnes_pop', etc. dans votre script")
print("3. Vérifiez que les séparateurs sont corrects")
print("4. Assurez-vous que les colonnes d'ID et d'année sont cohérentes entre les datasets")
print("5. Relancez le script principal avec les bons noms de colonnes")