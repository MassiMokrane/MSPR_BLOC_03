import pandas as pd
import unicodedata
import re

def nettoyer_texte(texte):
    """
    Nettoie un texte en supprimant les accents, espaces multiples et caractères spéciaux
    """
    if pd.isna(texte):
        return texte
    
    # Convertir en string si ce n'est pas déjà le cas
    texte = str(texte)
    
    # Supprimer les accents
    texte = unicodedata.normalize('NFD', texte)
    texte = ''.join(char for char in texte if unicodedata.category(char) != 'Mn')
    
    # Remplacer les espaces par des underscores et nettoyer
    texte = re.sub(r'\s+', '_', texte.strip())
    
    # Supprimer les caractères spéciaux sauf underscores et tirets
    texte = re.sub(r'[^\w\-]', '', texte)
    
    return texte

def nettoyer_orientation_politique(orientation):
    """
    Nettoie spécifiquement la colonne orientation politique
    """
    if pd.isna(orientation):
        return orientation
    
    # Dictionnaire de mapping pour standardiser les orientations
    mapping_orientations = {
        'extreme_droite': 'ED',
        'extreme_gauche': 'EG',
        'centre': 'CE',
        'droite': 'D',
        'gauche': 'G'
    }
    
    # Nettoyer d'abord le texte
    orientation_clean = nettoyer_texte(orientation).lower()
    
    # Appliquer le mapping si trouvé, sinon retourner la version nettoyée avec première lettre majuscule
    return mapping_orientations.get(orientation_clean, orientation_clean.capitalize())

# Chemin du fichier
fichier2 = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR.csv"

try:
    # Lire le fichier CSV
    print("Lecture du fichier CSV...")
    df = pd.read_csv(fichier2)
    
    print(f"Fichier lu avec succès. Dimensions: {df.shape}")
    print(f"Colonnes originales: {list(df.columns)}")
    
    # Nettoyer les noms de colonnes
    print("\nNettoyage des noms de colonnes...")
    colonnes_originales = df.columns.tolist()
    colonnes_nettoyees = [nettoyer_texte(col) for col in colonnes_originales]
    
    # Créer un dictionnaire de mapping pour renommer
    mapping_colonnes = dict(zip(colonnes_originales, colonnes_nettoyees))
    df = df.rename(columns=mapping_colonnes)
    
    print(f"Colonnes après nettoyage: {list(df.columns)}")
    
    # Nettoyer la colonne Orientation Politique
    if 'Orientation_Politique' in df.columns:
        print("\nNettoyage de la colonne Orientation_Politique...")
        print(f"Valeurs uniques avant nettoyage: {df['Orientation_Politique'].unique()}")
        
        df['Orientation_Politique'] = df['Orientation_Politique'].apply(nettoyer_orientation_politique)
        
        print(f"Valeurs uniques après nettoyage: {df['Orientation_Politique'].unique()}")
    else:
        print("Attention: La colonne 'Orientation_Politique' n'a pas été trouvée après le nettoyage des colonnes.")
        print("Colonnes disponibles:", list(df.columns))
    
    # Sauvegarder le fichier nettoyé
    fichier_sortie = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/finale_2025.csv"
    
    print(f"\nSauvegarde du fichier nettoyé: {fichier_sortie}")
    df.to_csv(fichier_sortie, index=False, encoding='utf-8')
    
    print("✅ Nettoyage terminé avec succès!")
    print(f"📊 Fichier sauvegardé: finale_2025.csv")
    print(f"📈 Nombre de lignes: {len(df)}")
    print(f"📋 Nombre de colonnes: {len(df.columns)}")
    
    # Afficher un aperçu du résultat
    print("\n--- Aperçu du fichier nettoyé ---")
    print(df.head())
    
    # Afficher le mapping des colonnes
    print("\n--- Mapping des colonnes ---")
    for ancien, nouveau in mapping_colonnes.items():
        if ancien != nouveau:
            print(f"'{ancien}' → '{nouveau}'")

except FileNotFoundError:
    print(f"❌ Erreur: Le fichier {fichier2} n'existe pas.")
    print("Vérifiez le chemin du fichier.")
except Exception as e:
    print(f"❌ Erreur lors du traitement: {str(e)}")
    print("Vérifiez que le fichier est accessible et au bon format.")