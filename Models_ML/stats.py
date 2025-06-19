import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# 📊 Chargement des données
print("📊 Chargement des données...")
file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
data = pd.read_csv(file_path, sep=';')
print(f"Dataset complet: {data.shape[0]} lignes")

# 🗺️ Définition des régions françaises
regions_francaises = {
    'Auvergne-Rhône-Alpes': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
    'Bourgogne-Franche-Comté': ['21', '25', '39', '58', '70', '71', '89', '90'],
    'Bretagne': ['22', '29', '35', '56'],
    'Centre-Val de Loire': ['18', '28', '36', '37', '41', '45'],
    'Corse': ['2A', '2B'],
    'Grand Est': ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
    'Hauts-de-France': ['02', '59', '60', '62', '80'],
    'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
    'Normandie': ['14', '27', '50', '61', '76'],
    'Nouvelle-Aquitaine': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
    'Occitanie': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
    'Pays de la Loire': ['44', '49', '53', '72', '85'],
    'Provence-Alpes-Côte d\'Azur': ['04', '05', '06', '13', '83', '84']
}

# 🏛️ Noms des départements pour référence
dept_names = {
    '01': 'Ain', '02': 'Aisne', '03': 'Allier', '04': 'Alpes-de-Haute-Provence', '05': 'Hautes-Alpes',
    '06': 'Alpes-Maritimes', '07': 'Ardèche', '08': 'Ardennes', '09': 'Ariège', '10': 'Aube',
    '11': 'Aude', '12': 'Aveyron', '13': 'Bouches-du-Rhône', '14': 'Calvados', '15': 'Cantal',
    '16': 'Charente', '17': 'Charente-Maritime', '18': 'Cher', '19': 'Corrèze', '21': 'Côte-d\'Or',
    '22': 'Côtes-d\'Armor', '23': 'Creuse', '24': 'Dordogne', '25': 'Doubs', '26': 'Drôme',
    '27': 'Eure', '28': 'Eure-et-Loir', '29': 'Finistère', '30': 'Gard', '31': 'Haute-Garonne',
    '32': 'Gers', '33': 'Gironde', '34': 'Hérault', '35': 'Ille-et-Vilaine', '36': 'Indre',
    '37': 'Indre-et-Loire', '38': 'Isère', '39': 'Jura', '40': 'Landes', '41': 'Loir-et-Cher',
    '42': 'Loire', '43': 'Haute-Loire', '44': 'Loire-Atlantique', '45': 'Loiret', '46': 'Lot',
    '47': 'Lot-et-Garonne', '48': 'Lozère', '49': 'Maine-et-Loire', '50': 'Manche', '51': 'Marne',
    '52': 'Haute-Marne', '53': 'Mayenne', '54': 'Meurthe-et-Moselle', '55': 'Meuse', '56': 'Morbihan',
    '57': 'Moselle', '58': 'Nièvre', '59': 'Nord', '60': 'Oise', '61': 'Orne', '62': 'Pas-de-Calais',
    '63': 'Puy-de-Dôme', '64': 'Pyrénées-Atlantiques', '65': 'Hautes-Pyrénées', '66': 'Pyrénées-Orientales',
    '67': 'Bas-Rhin', '68': 'Haut-Rhin', '69': 'Rhône', '70': 'Haute-Saône', '71': 'Saône-et-Loire',
    '72': 'Sarthe', '73': 'Savoie', '74': 'Haute-Savoie', '75': 'Paris', '76': 'Seine-Maritime',
    '77': 'Seine-et-Marne', '78': 'Yvelines', '79': 'Deux-Sèvres', '80': 'Somme', '81': 'Tarn',
    '82': 'Tarn-et-Garonne', '83': 'Var', '84': 'Vaucluse', '85': 'Vendée', '86': 'Vienne',
    '87': 'Haute-Vienne', '88': 'Vosges', '89': 'Yonne', '90': 'Territoire de Belfort',
    '91': 'Essonne', '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne',
    '95': 'Val-d\'Oise', '2A': 'Corse-du-Sud', '2B': 'Haute-Corse'
}

# 🎯 Mapping des orientations politiques pour plus de clarté
orientation_labels = {
    0: 'Extrême gauche',
    1: 'Gauche',
    2: 'Centre',
    3: 'Droite',
    4: 'Extrême droite'
}

# 🔧 Préparation des données
print("\n🔧 Préparation des données...")
data['code_dept'] = data['commune_id'].astype(str).str[:2]

# Gestion spéciale pour la Corse
mask_corse = data['code_dept'] == '20'
if mask_corse.any():
    # Pour la Corse, utiliser les 3 premiers caractères pour distinguer 2A et 2B
    data.loc[mask_corse, 'code_dept'] = data.loc[mask_corse, 'commune_id'].astype(str).str[:2]
    # Si c'est 20, on regarde le 3ème caractère
    mask_20 = data['code_dept'] == '20'
    if mask_20.any():
        data.loc[mask_20, 'code_dept'] = '2A'  # Par défaut, ajuster selon vos données

# Nettoyer les données manquantes
data_clean = data.dropna(subset=['orientation_politique']).copy()
print(f"Données nettoyées: {data_clean.shape[0]} lignes")

# 📊 Analyse par région
results = []
regional_stats = {}

print("\n" + "="*80)
print("📊 ANALYSE DES ORIENTATIONS POLITIQUES PAR RÉGION")
print("="*80)

for region_name, departements in regions_francaises.items():
    print(f"\n🏛️ {region_name.upper()}")
    print("-" * 50)
    
    # Filtrer les données de la région
    region_data = data_clean[data_clean['code_dept'].isin(departements)].copy()
    
    if len(region_data) == 0:
        print("   ⚠️  Aucune donnée disponible pour cette région")
        continue
    
    total_communes = len(region_data)
    print(f"📍 Nombre total de communes: {total_communes}")
    
    # Afficher les départements présents
    depts_presents = sorted(region_data['code_dept'].unique())
    print(f"🗂️  Départements analysés: {', '.join(depts_presents)}")
    
    for dept in depts_presents:
        nb_communes_dept = len(region_data[region_data['code_dept'] == dept])
        dept_name = dept_names.get(dept, f"Département {dept}")
        print(f"   • {dept} - {dept_name}: {nb_communes_dept} communes")
    
    # Calculer les statistiques d'orientation
    orientation_counts = region_data['orientation_politique'].value_counts().sort_index()
    
    print(f"\n📊 Distribution des orientations politiques:")
    region_stats = {}
    
    for orientation in sorted(orientation_counts.index):
        count = orientation_counts[orientation]
        percentage = (count / total_communes) * 100
        label = orientation_labels.get(orientation, f"Orientation {orientation}")
        
        print(f"   • {label:<15} (classe {orientation}): {count:>4} communes ({percentage:>5.1f}%)")
        
        region_stats[orientation] = {
            'count': count,
            'percentage': percentage,
            'label': label
        }
        
        # Ajouter aux résultats globaux
        results.append({
            'Region': region_name,
            'Orientation': orientation,
            'Label': label,
            'Nombre': count,
            'Pourcentage': percentage,
            'Total_Communes': total_communes
        })
    
    # Identifier l'orientation dominante
    orientation_dominante = orientation_counts.idxmax()
    max_percentage = (orientation_counts.max() / total_communes) * 100
    label_dominante = orientation_labels.get(orientation_dominante, f"Orientation {orientation_dominante}")
    
    print(f"🏆 Orientation dominante: {label_dominante} ({max_percentage:.1f}%)")
    
    regional_stats[region_name] = {
        'total_communes': total_communes,
        'departements': depts_presents,
        'orientations': region_stats,
        'dominante': {
            'orientation': orientation_dominante,
            'label': label_dominante,
            'percentage': max_percentage
        }
    }

# 📈 Création des visualisations
print("\n📈 Génération des graphiques...")

# Convertir en DataFrame pour les visualisations
df_results = pd.DataFrame(results)

if len(df_results) > 0:
    # 1. Graphique en barres empilées par région
    plt.figure(figsize=(20, 12))
    
    # Préparer les données pour le graphique empilé
    pivot_data = df_results.pivot(index='Region', columns='Label', values='Pourcentage').fillna(0)
    
    # Graphique principal
    plt.subplot(2, 2, 1)
    ax1 = pivot_data.plot(kind='bar', stacked=True, 
                         colormap='Set3', 
                         figsize=(15, 8))
    plt.title('Distribution des orientations politiques par région (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Région')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Orientation politique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Heatmap des pourcentages
    plt.subplot(2, 2, 2)
    sns.heatmap(pivot_data.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Pourcentage (%)'})
    plt.title('Heatmap des orientations par région')
    plt.xlabel('Région')
    plt.ylabel('Orientation politique')
    
    # 3. Nombre total de communes par région
    plt.subplot(2, 2, 3)
    communes_by_region = df_results.groupby('Region')['Total_Communes'].first().sort_values(ascending=False)
    communes_by_region.plot(kind='bar', color='skyblue')
    plt.title('Nombre de communes par région')
    plt.xlabel('Région')
    plt.ylabel('Nombre de communes')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 4. Distribution globale des orientations
    plt.subplot(2, 2, 4)
    global_orientation = df_results.groupby('Label')['Nombre'].sum().sort_values(ascending=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(global_orientation)))
    wedges, texts, autotexts = plt.pie(global_orientation.values, 
                                      labels=global_orientation.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    plt.title('Distribution globale des orientations\n(toutes régions confondues)')
    
    plt.tight_layout()
    plt.show()
    
    # 📊 Tableau récapitulatif
    print("\n📊 TABLEAU RÉCAPITULATIF:")
    print("=" * 100)
    
    summary_table = []
    for region_name, stats in regional_stats.items():
        summary_table.append({
            'Région': region_name,
            'Communes': stats['total_communes'],
            'Départements': len(stats['departements']),
            'Orientation dominante': stats['dominante']['label'],
            'Pourcentage dominant': f"{stats['dominante']['percentage']:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_table)
    summary_df = summary_df.sort_values('Communes', ascending=False)
    
    print(summary_df.to_string(index=False))
    
    # 🏆 Top 5 des régions par orientation
    print("\n🏆 TOP 5 DES RÉGIONS PAR ORIENTATION:")
    print("=" * 60)
    
    for orientation_label in orientation_labels.values():
        if orientation_label in df_results['Label'].values:
            top_regions = df_results[df_results['Label'] == orientation_label].nlargest(5, 'Pourcentage')
            if len(top_regions) > 0:
                print(f"\n🎯 {orientation_label}:")
                for _, row in top_regions.iterrows():
                    print(f"   • {row['Region']:<25}: {row['Pourcentage']:>5.1f}% ({row['Nombre']} communes)")

else:
    print("⚠️ Aucune donnée disponible pour générer les visualisations")

print(f"\n🎉 Analyse terminée! {len(regional_stats)} régions analysées.")
print("🚀 Toutes les statistiques et visualisations ont été générées avec succès!")