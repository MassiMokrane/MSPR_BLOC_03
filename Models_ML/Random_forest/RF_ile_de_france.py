
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # 📊 Chargement des données
# # print("📊 Chargement des données...")
# # file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
# # data = pd.read_csv(file_path, sep=';')
# # print(f"Dataset complet: {data.shape[0]} lignes")

# # # 🗺️ Filtrage sur l'Île-de-France
# # idf_depts = ['75', '77', '78', '91', '92', '93', '94', '95']
# # data['code_dept'] = data['commune_id'].astype(str).str[:2]
# # data_region = data[data['code_dept'].isin(idf_depts)].copy()
# # print(f"📍 Région sélectionnée : Île-de-France ({', '.join(idf_depts)})")
# # print(f"✅ Nombre de communes : {data_region.shape[0]}")

# # # 🎯 Vérification des classes
# # print("\nClasses d'orientation politique :")
# # print(data_region['orientation_politique'].value_counts().sort_index())

# # # 🔧 Nettoyage des données
# # data_region = data_region.dropna(subset=['orientation_politique'])

# # # 📋 Sélection des variables (features)
# # features_selected = [
# #     'nb_inscrits',
# #     'nb_abstentions',
# #     'pct_population_senior',
# #     'pct_population_sans_activite',
# #     'pct_population_etrangere',
# #     'taux_chomage_pct',
# #     'nb_population_active',
# #     'Population_Totale',
# #     'nb_crimes'
# # ]
# # X = data_region[features_selected].copy()
# # y = data_region['orientation_politique'].copy()

# # # 🔄 Remplir les valeurs manquantes avec la médiane
# # for col in X.columns:
# #     if X[col].isnull().sum() > 0:
# #         X[col] = X[col].fillna(X[col].median())

# # # 📦 Séparation train/test
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, stratify=y, random_state=42
# # )

# # # 📏 Standardisation
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # 🌲 Entraînement Random Forest
# # rf_model = RandomForestClassifier(
# #     n_estimators=200,
# #     max_depth=15,
# #     min_samples_split=5,
# #     min_samples_leaf=2,
# #     random_state=42,
# #     n_jobs=-1
# # )
# # rf_model.fit(X_train_scaled, y_train)

# # # 🎯 Prédictions et évaluation
# # y_pred = rf_model.predict(X_test_scaled)
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"\n🏆 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
# # print("\n📊 Rapport de classification :")
# # print(classification_report(y_test, y_pred))

# # # 🔲 Matrice de confusion
# # plt.figure(figsize=(8, 6))
# # cm = confusion_matrix(y_test, y_pred)
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# # plt.title("Matrice de confusion - Orientation Politique")
# # plt.xlabel("Prédit")
# # plt.ylabel("Réel")
# # plt.tight_layout()
# # plt.show()

# # # 🔍 Importance des features
# # feature_importance = pd.DataFrame({
# #     'feature': features_selected,
# #     'importance': rf_model.feature_importances_
# # }).sort_values(by='importance', ascending=False)

# # print("\n🔍 Importance des features :")
# # for i, row in feature_importance.iterrows():
# #     print(f"{row['feature']:<30} : {row['importance']:.4f}")

# # plt.figure(figsize=(10, 6))
# # sns.barplot(data=feature_importance, x='importance', y='feature')
# # plt.title("Importance des variables")
# # plt.xlabel("Importance")
# # plt.tight_layout()
# # plt.show()

# # # 🔮 Fonction de prédiction personnalisée
# # def predire_orientation(nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
# #                         pct_sans_activite, pct_etrangere, 
# #                         taux_chomage, population_totale,nb_crimes):
# #     new_data = np.array([[nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
# #                           pct_sans_activite, pct_etrangere, 
# #                           taux_chomage, population_totale,nb_crimes]])
# #     new_data_scaled = scaler.transform(new_data)
# #     prediction = rf_model.predict(new_data_scaled)[0]
# #     proba = rf_model.predict_proba(new_data_scaled)[0]
# #     return prediction, proba

# # # 🧪 Test de la fonction
# # print("\n🔮 TEST DE PRÉDICTION :")
# # test_prediction, test_proba = predire_orientation(
# #     nb_inscrits=900,
# #     nb_abstentions=150,
# #     pct_jeune=26.0,
# #     pct_senior=18.0,
# #     pct_sans_activite=22.0,
# #     pct_etrangere=11.0,
# #     taux_chomage=0.10,
# #     population_totale=1300,
# #     nb_crimes=0,
# # )

# # print(f"Orientation prédite : {test_prediction}")
# # print("Probabilités associées :")
# # for classe, proba in zip(sorted(y.unique()), test_proba):
# #     print(f"  Classe {classe}: {proba:.3f}")

# # # 📝 Résumé
# # print("\n🎉 RÉSUMÉ FINAL :")
# # print(f"📍 Région analysée : Île-de-France ({', '.join(idf_depts)})")
# # print(f"📊 Communes utilisées : {len(data_region)}")
# # print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
# # print(f"🌲 Nombre de features : {len(features_selected)}")

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, validation_curve, learning_curve
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 📊 Chargement des données
# print("📊 Chargement des données...")
# file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
# data = pd.read_csv(file_path, sep=';')
# print(f"Dataset complet: {data.shape[0]} lignes")

# # 🗺️ Filtrage sur la région Île-de-France
# ile_de_france_depts = ['75', '77', '78', '91', '92', '93', '94', '95']
# data['code_dept'] = data['commune_id'].astype(str).str[:2]
# data_region = data[data['code_dept'].isin(ile_de_france_depts)].copy()

# print(f"📍 Région sélectionnée : Île-de-France")
# print(f"🏛️ Départements inclus : {', '.join(ile_de_france_depts)}")
# print(f"📍 Départements détaillés :")
# dept_names = {
#     '75': 'Paris', '77': 'Seine-et-Marne', '78': 'Yvelines', 
#     '91': 'Essonne', '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis',
#     '94': 'Val-de-Marne', '95': 'Val-d\'Oise'
# }
# for code, nom in dept_names.items():
#     nb_communes = len(data_region[data_region['code_dept'] == code])
#     print(f"   {code} - {nom}: {nb_communes} communes")

# print(f"✅ Nombre total de communes : {data_region.shape[0]}")

# # 🎯 Vérification des classes
# print("\nClasses d'orientation politique :")
# print(data_region['orientation_politique'].value_counts().sort_index())

# # 🔧 Nettoyage des données
# data_region = data_region.dropna(subset=['orientation_politique'])

# # 📋 Sélection des variables (features)
# features_selected = [
#     'nb_inscrits',
#     'nb_abstentions',
#     'pct_population_senior',
#     'pct_population_sans_activite',
#     'pct_population_etrangere',
#     'taux_chomage_pct',
#     'nb_population_active',
#     'Population_Totale'
# ]
# X = data_region[features_selected].copy()
# y = data_region['orientation_politique'].copy()

# # 🔄 Remplir les valeurs manquantes avec la médiane
# print("\n🔄 Traitement des valeurs manquantes...")
# for col in X.columns:
#     nb_missing = X[col].isnull().sum()
#     if nb_missing > 0:
#         print(f"   {col}: {nb_missing} valeurs manquantes → remplacées par la médiane")
#         X[col] = X[col].fillna(X[col].median())

# # 📦 Séparation train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# print(f"\n📦 Séparation des données :")
# print(f"   Training: {X_train.shape[0]} communes")
# print(f"   Test: {X_test.shape[0]} communes")

# # 📏 Standardisation
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print("\n📏 Standardisation appliquée aux features")

# # 🌲 Entraînement Random Forest
# print("\n🌲 Entraînement du modèle Random Forest...")
# rf_model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=15,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     random_state=42,
#     n_jobs=-1
# )
# rf_model.fit(X_train_scaled, y_train)

# # 🎯 Prédictions et évaluation
# y_pred = rf_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\n🏆 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
# print("\n📊 Rapport de classification :")
# print(classification_report(y_test, y_pred))

# # 📈 COURBES D'APPRENTISSAGE (équivalent aux courbes de perte pour Random Forest)
# print("\n📈 Calcul des courbes d'apprentissage...")
# train_sizes, train_scores, val_scores = learning_curve(
#     RandomForestClassifier(n_estimators=100, random_state=42),
#     X_train_scaled, y_train,
#     cv=5,
#     train_sizes=np.linspace(0.1, 1.0, 10),
#     n_jobs=-1,
#     random_state=42
# )

# # Calcul des moyennes et écarts-types
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# val_mean = np.mean(val_scores, axis=1)
# val_std = np.std(val_scores, axis=1)

# plt.figure(figsize=(15, 10))

# # Courbe d'apprentissage (équivalent aux courbes de précision)
# plt.subplot(2, 2, 1)
# plt.plot(train_sizes, train_mean, label='Train', linewidth=2, marker='o', color='#2E86AB')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2E86AB')
# plt.plot(train_sizes, val_mean, label='Validation', linewidth=2, marker='s', color='#A23B72')
# plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#A23B72')
# plt.xlabel('Taille du dataset d\'entraînement')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Courbes d\'apprentissage - Île-de-France')
# plt.grid(True, alpha=0.3)

# # 📊 COURBE DE VALIDATION (hyperparamètre n_estimators)
# print("📊 Calcul de la courbe de validation...")
# n_estimators_range = range(10, 201, 20)
# train_scores_val, val_scores_val = validation_curve(
#     RandomForestClassifier(random_state=42),
#     X_train_scaled, y_train,
#     param_name='n_estimators',
#     param_range=n_estimators_range,
#     cv=5,
#     n_jobs=-1
# )

# train_mean_val = np.mean(train_scores_val, axis=1)
# train_std_val = np.std(train_scores_val, axis=1)
# val_mean_val = np.mean(val_scores_val, axis=1)
# val_std_val = np.std(val_scores_val, axis=1)

# plt.subplot(2, 2, 2)
# plt.plot(n_estimators_range, train_mean_val, label='Train', linewidth=2, marker='o', color='#2E86AB')
# plt.fill_between(n_estimators_range, train_mean_val - train_std_val, 
#                  train_mean_val + train_std_val, alpha=0.2, color='#2E86AB')
# plt.plot(n_estimators_range, val_mean_val, label='Validation', linewidth=2, marker='s', color='#A23B72')
# plt.fill_between(n_estimators_range, val_mean_val - val_std_val, 
#                  val_mean_val + val_std_val, alpha=0.2, color='#A23B72')
# plt.xlabel('Nombre d\'estimateurs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Courbe de validation (n_estimators)')
# plt.grid(True, alpha=0.3)

# # 📊 COURBE DE VALIDATION (max_depth)
# print("📊 Calcul de la courbe de validation pour max_depth...")
# max_depth_range = range(5, 26, 2)
# train_scores_depth, val_scores_depth = validation_curve(
#     RandomForestClassifier(n_estimators=100, random_state=42),
#     X_train_scaled, y_train,
#     param_name='max_depth',
#     param_range=max_depth_range,
#     cv=5,
#     n_jobs=-1
# )

# train_mean_depth = np.mean(train_scores_depth, axis=1)
# train_std_depth = np.std(train_scores_depth, axis=1)
# val_mean_depth = np.mean(val_scores_depth, axis=1)
# val_std_depth = np.std(val_scores_depth, axis=1)

# plt.subplot(2, 2, 3)
# plt.plot(max_depth_range, train_mean_depth, label='Train', linewidth=2, marker='o', color='#2E86AB')
# plt.fill_between(max_depth_range, train_mean_depth - train_std_depth, 
#                  train_mean_depth + train_std_depth, alpha=0.2, color='#2E86AB')
# plt.plot(max_depth_range, val_mean_depth, label='Validation', linewidth=2, marker='s', color='#A23B72')
# plt.fill_between(max_depth_range, val_mean_depth - val_std_depth, 
#                  val_mean_depth + val_std_depth, alpha=0.2, color='#A23B72')
# plt.xlabel('Profondeur maximum')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Courbe de validation (max_depth)')
# plt.grid(True, alpha=0.3)

# # 📊 Distribution des classes par département
# dept_class_data = []
# for dept in ile_de_france_depts:
#     dept_data = data_region[data_region['code_dept'] == dept]
#     if len(dept_data) > 0:
#         for orientation in dept_data['orientation_politique'].unique():
#             count = len(dept_data[dept_data['orientation_politique'] == orientation])
#             dept_class_data.append({
#                 'Département': f"{dept} ({dept_names[dept][:15]})",
#                 'Orientation': orientation,
#                 'Nombre': count
#             })

# dept_class_df = pd.DataFrame(dept_class_data)
# plt.subplot(2, 2, 4)
# if len(dept_class_df) > 0:
#     pivot_data = dept_class_df.pivot(index='Département', columns='Orientation', values='Nombre').fillna(0)
#     pivot_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Set3')
#     plt.title('Distribution des orientations par département')
#     plt.xlabel('Département')
#     plt.ylabel('Nombre de communes')
#     plt.xticks(rotation=45, ha='right')
#     plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.show()

# # 🔲 Matrice de confusion
# plt.figure(figsize=(10, 8))
# cm = confusion_matrix(y_test, y_pred)
# class_names = sorted(y.unique())
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=class_names, yticklabels=class_names)
# plt.title("Matrice de confusion - Orientation Politique (Île-de-France)")
# plt.xlabel("Prédit")
# plt.ylabel("Réel")
# plt.tight_layout()
# plt.show()

# # 🔍 Importance des features
# feature_importance = pd.DataFrame({
#     'feature': features_selected,
#     'importance': rf_model.feature_importances_
# }).sort_values(by='importance', ascending=False)

# print("\n🔍 Importance des features :")
# for i, row in feature_importance.iterrows():
#     print(f"{row['feature']:<30} : {row['importance']:.4f}")

# plt.figure(figsize=(12, 8))
# colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']
# sns.barplot(data=feature_importance, x='importance', y='feature', 
#             palette=colors[:len(feature_importance)])
# plt.title("Importance des variables - Île-de-France")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()

# # 📊 Performance par classe
# plt.figure(figsize=(12, 8))
# report_dict = classification_report(y_test, y_pred, output_dict=True)
# classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
# metrics = ['precision', 'recall', 'f1-score']

# performance_data = []
# for classe in classes:
#     for metric in metrics:
#         performance_data.append({
#             'Classe': classe,
#             'Métrique': metric,
#             'Score': report_dict[classe][metric]
#         })

# performance_df = pd.DataFrame(performance_data)
# sns.barplot(data=performance_df, x='Classe', y='Score', hue='Métrique', palette='Set2')
# plt.title('Performance par classe politique - Île-de-France')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 🔮 Fonction de prédiction personnalisée
# def predire_orientation_ile_de_france(nb_inscrits, nb_abstentions, pct_senior,
#                                       pct_sans_activite, pct_etrangere, taux_chomage,
#                                       nb_population_active, population_totale):
#     """
#     Prédire l'orientation politique d'une commune en Île-de-France
    
#     Paramètres:
#     - nb_inscrits: Nombre d'inscrits sur les listes électorales
#     - nb_abstentions: Nombre d'abstentions
#     - pct_senior: Pourcentage de population senior
#     - pct_sans_activite: Pourcentage de population sans activité
#     - pct_etrangere: Pourcentage de population étrangère
#     - taux_chomage: Taux de chômage (%)
#     - nb_population_active: Nombre de population active
#     - population_totale: Population totale
#     """
#     new_data = np.array([[nb_inscrits, nb_abstentions, pct_senior,
#                           pct_sans_activite, pct_etrangere, taux_chomage,
#                           nb_population_active, population_totale]])
#     new_data_scaled = scaler.transform(new_data)
#     prediction = rf_model.predict(new_data_scaled)[0]
#     proba = rf_model.predict_proba(new_data_scaled)[0]
    
#     # Créer un dictionnaire avec les classes et leurs probabilités
#     classes = rf_model.classes_
#     proba_dict = {classes[i]: proba[i] for i in range(len(proba))}
    
#     return prediction, proba_dict

# # 📊 Analyse des statistiques régionales
# print("\n📊 STATISTIQUES DESCRIPTIVES - ÎLE-DE-FRANCE:")
# print("="*60)
# for feature in features_selected:
#     mean_val = data_region[feature].mean()
#     median_val = data_region[feature].median()
#     std_val = data_region[feature].std()
#     print(f"{feature}:")
#     print(f"   Moyenne: {mean_val:.2f} | Médiane: {median_val:.2f} | Écart-type: {std_val:.2f}")

# # 📝 Résumé final
# print("\n🎉 RÉSUMÉ FINAL - ÎLE-DE-FRANCE:")
# print("="*60)
# print(f"📍 Région analysée : Île-de-France")
# print(f"🏛️ Départements : {len(ile_de_france_depts)} départements")
# print(f"📊 Communes utilisées : {len(data_region)}")
# print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
# print(f"🌲 Nombre de features : {len(features_selected)}")
# print(f"📈 Courbes générées : Apprentissage + Validation + Distribution")
# print(f"🏆 Meilleure feature : {feature_importance.iloc[0]['feature']}")

# # 💡 Exemple d'utilisation
# print("\n💡 EXEMPLE DE PRÉDICTION:")
# print("Prédiction pour une commune type d'Île-de-France...")
# example_prediction, example_proba = predire_orientation_ile_de_france(
#     nb_inscrits=2500,
#     nb_abstentions=750,
#     pct_senior=22.0,
#     pct_sans_activite=15.0,
#     pct_etrangere=12.5,
#     taux_chomage=8.5,
#     nb_population_active=1800,
#     population_totale=3200
# )
# print(f"🔮 Orientation prédite : {example_prediction}")
# print("📊 Probabilités par classe :")
# for classe, prob in sorted(example_proba.items(), key=lambda x: x[1], reverse=True):
#     print(f"   {classe}: {prob:.3f} ({prob*100:.1f}%)")

# print("\n🎯 Modèle prêt pour des prédictions sur l'Île-de-France ! 🚀")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📊 Chargement des données
print("📊 Chargement des données...")
file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
data = pd.read_csv(file_path, sep=';')
print(f"Dataset complet: {data.shape[0]} lignes")

# 🗺️ Filtrage sur la région Île-de-France
ile_de_france_depts = ['75', '77', '78', '91', '92', '93', '94', '95']
data['code_dept'] = data['commune_id'].astype(str).str[:2]
data_region = data[data['code_dept'].isin(ile_de_france_depts)].copy()

print(f"📍 Région sélectionnée : Île-de-France")
print(f"🏛️ Départements inclus : {', '.join(ile_de_france_depts)}")
print(f"📍 Départements détaillés :")
dept_names = {
    '75': 'Paris', '77': 'Seine-et-Marne', '78': 'Yvelines', 
    '91': 'Essonne', '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis',
    '94': 'Val-de-Marne', '95': 'Val-d\'Oise'
}
for code, nom in dept_names.items():
    nb_communes = len(data_region[data_region['code_dept'] == code])
    print(f"   {code} - {nom}: {nb_communes} communes")

print(f"✅ Nombre total de communes : {data_region.shape[0]}")

# 🎯 Vérification des classes
print("\nClasses d'orientation politique :")
print(data_region['orientation_politique'].value_counts().sort_index())

# 🔧 Nettoyage des données
data_region = data_region.dropna(subset=['orientation_politique'])

# 📋 Sélection des variables (features)
features_selected = [
    'nb_inscrits',
    'nb_abstentions',
    'pct_population_senior',
    'pct_population_sans_activite',
    'pct_population_etrangere',
    'taux_chomage_pct',
    'nb_population_active',
    'Population_Totale'
]
X = data_region[features_selected].copy()
y = data_region['orientation_politique'].copy()

# 🔄 Remplir les valeurs manquantes avec la médiane
print("\n🔄 Traitement des valeurs manquantes...")
for col in X.columns:
    nb_missing = X[col].isnull().sum()
    if nb_missing > 0:
        print(f"   {col}: {nb_missing} valeurs manquantes → remplacées par la médiane")
        X[col] = X[col].fillna(X[col].median())

# 📦 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n📦 Séparation des données :")
print(f"   Training: {X_train.shape[0]} communes")
print(f"   Test: {X_test.shape[0]} communes")

# 📏 Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n📏 Standardisation appliquée aux features")

# 🌲 Entraînement Random Forest - Paramètres optimisés contre l'overfitting
print("\n🌲 Entraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,        # Réduit de 200 à 100 (suffisant d'après la courbe)
    max_depth=8,             # Réduit de 15 à 8 pour limiter l'overfitting
    min_samples_split=10,    # Augmenté de 5 à 10
    min_samples_leaf=5,      # Augmenté de 2 à 5
    max_features='sqrt',     # Limitation des features par arbre
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# 🎯 Prédictions et évaluation
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🏆 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# 📈 COURBES D'APPRENTISSAGE (équivalent aux courbes de perte pour Random Forest)
print("\n📈 Calcul des courbes d'apprentissage...")
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    random_state=42
)

# Calcul des moyennes et écarts-types
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(15, 10))

# Courbe d'apprentissage (équivalent aux courbes de précision)
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Train', linewidth=2, marker='o', color='#2E86AB')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2E86AB')
plt.plot(train_sizes, val_mean, label='Validation', linewidth=2, marker='s', color='#A23B72')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#A23B72')
plt.xlabel('Taille du dataset d\'entraînement')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbes d\'apprentissage - Île-de-France')
plt.grid(True, alpha=0.3)

# 📊 COURBE DE VALIDATION (hyperparamètre n_estimators)
print("📊 Calcul de la courbe de validation...")
n_estimators_range = range(10, 201, 20)
train_scores_val, val_scores_val = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train_scaled, y_train,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=5,
    n_jobs=-1
)

train_mean_val = np.mean(train_scores_val, axis=1)
train_std_val = np.std(train_scores_val, axis=1)
val_mean_val = np.mean(val_scores_val, axis=1)
val_std_val = np.std(val_scores_val, axis=1)

plt.subplot(2, 2, 2)
plt.plot(n_estimators_range, train_mean_val, label='Train', linewidth=2, marker='o', color='#2E86AB')
plt.fill_between(n_estimators_range, train_mean_val - train_std_val, 
                 train_mean_val + train_std_val, alpha=0.2, color='#2E86AB')
plt.plot(n_estimators_range, val_mean_val, label='Validation', linewidth=2, marker='s', color='#A23B72')
plt.fill_between(n_estimators_range, val_mean_val - val_std_val, 
                 val_mean_val + val_std_val, alpha=0.2, color='#A23B72')
plt.xlabel('Nombre d\'estimateurs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (n_estimators)')
plt.grid(True, alpha=0.3)

# 📊 COURBE DE VALIDATION (max_depth)
print("📊 Calcul de la courbe de validation pour max_depth...")
max_depth_range = range(5, 26, 2)
train_scores_depth, val_scores_depth = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train,
    param_name='max_depth',
    param_range=max_depth_range,
    cv=5,
    n_jobs=-1
)

train_mean_depth = np.mean(train_scores_depth, axis=1)
train_std_depth = np.std(train_scores_depth, axis=1)
val_mean_depth = np.mean(val_scores_depth, axis=1)
val_std_depth = np.std(val_scores_depth, axis=1)

plt.subplot(2, 2, 3)
plt.plot(max_depth_range, train_mean_depth, label='Train', linewidth=2, marker='o', color='#2E86AB')
plt.fill_between(max_depth_range, train_mean_depth - train_std_depth, 
                 train_mean_depth + train_std_depth, alpha=0.2, color='#2E86AB')
plt.plot(max_depth_range, val_mean_depth, label='Validation', linewidth=2, marker='s', color='#A23B72')
plt.fill_between(max_depth_range, val_mean_depth - val_std_depth, 
                 val_mean_depth + val_std_depth, alpha=0.2, color='#A23B72')
plt.xlabel('Profondeur maximum')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (max_depth)')
plt.grid(True, alpha=0.3)

# 📊 Distribution des classes par département
dept_class_data = []
for dept in ile_de_france_depts:
    dept_data = data_region[data_region['code_dept'] == dept]
    if len(dept_data) > 0:
        for orientation in dept_data['orientation_politique'].unique():
            count = len(dept_data[dept_data['orientation_politique'] == orientation])
            dept_class_data.append({
                'Département': f"{dept} ({dept_names[dept][:15]})",
                'Orientation': orientation,
                'Nombre': count
            })

dept_class_df = pd.DataFrame(dept_class_data)
plt.subplot(2, 2, 4)
if len(dept_class_df) > 0:
    pivot_data = dept_class_df.pivot(index='Département', columns='Orientation', values='Nombre').fillna(0)
    pivot_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Set3')
    plt.title('Distribution des orientations par département')
    plt.xlabel('Département')
    plt.ylabel('Nombre de communes')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# 🔲 Matrice de confusion
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
class_names = sorted(y.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matrice de confusion - Orientation Politique (Île-de-France)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# 🔍 Importance des features
feature_importance = pd.DataFrame({
    'feature': features_selected,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n🔍 Importance des features :")
for i, row in feature_importance.iterrows():
    print(f"{row['feature']:<30} : {row['importance']:.4f}")

plt.figure(figsize=(12, 8))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']
sns.barplot(data=feature_importance, x='importance', y='feature', 
            palette=colors[:len(feature_importance)])
plt.title("Importance des variables - Île-de-France")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 📊 Performance par classe
plt.figure(figsize=(12, 8))
report_dict = classification_report(y_test, y_pred, output_dict=True)
classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
metrics = ['precision', 'recall', 'f1-score']

performance_data = []
for classe in classes:
    for metric in metrics:
        performance_data.append({
            'Classe': classe,
            'Métrique': metric,
            'Score': report_dict[classe][metric]
        })

performance_df = pd.DataFrame(performance_data)
sns.barplot(data=performance_df, x='Classe', y='Score', hue='Métrique', palette='Set2')
plt.title('Performance par classe politique - Île-de-France')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔮 Fonction de prédiction personnalisée
def predire_orientation_ile_de_france(nb_inscrits, nb_abstentions, pct_senior,
                                      pct_sans_activite, pct_etrangere, taux_chomage,
                                      nb_population_active, population_totale):
    """
    Prédire l'orientation politique d'une commune en Île-de-France
    
    Paramètres:
    - nb_inscrits: Nombre d'inscrits sur les listes électorales
    - nb_abstentions: Nombre d'abstentions
    - pct_senior: Pourcentage de population senior
    - pct_sans_activite: Pourcentage de population sans activité
    - pct_etrangere: Pourcentage de population étrangère
    - taux_chomage: Taux de chômage (%)
    - nb_population_active: Nombre de population active
    - population_totale: Population totale
    """
    new_data = np.array([[nb_inscrits, nb_abstentions, pct_senior,
                          pct_sans_activite, pct_etrangere, taux_chomage,
                          nb_population_active, population_totale]])
    new_data_scaled = scaler.transform(new_data)
    prediction = rf_model.predict(new_data_scaled)[0]
    proba = rf_model.predict_proba(new_data_scaled)[0]
    
    # Créer un dictionnaire avec les classes et leurs probabilités
    classes = rf_model.classes_
    proba_dict = {classes[i]: proba[i] for i in range(len(proba))}
    
    return prediction, proba_dict

# 📊 Analyse des statistiques régionales
print("\n📊 STATISTIQUES DESCRIPTIVES - ÎLE-DE-FRANCE:")
print("="*60)
for feature in features_selected:
    mean_val = data_region[feature].mean()
    median_val = data_region[feature].median()
    std_val = data_region[feature].std()
    print(f"{feature}:")
    print(f"   Moyenne: {mean_val:.2f} | Médiane: {median_val:.2f} | Écart-type: {std_val:.2f}")

# 📝 Résumé final
print("\n🎉 RÉSUMÉ FINAL - ÎLE-DE-FRANCE:")
print("="*60)
print(f"📍 Région analysée : Île-de-France")
print(f"🏛️ Départements : {len(ile_de_france_depts)} départements")
print(f"📊 Communes utilisées : {len(data_region)}")
print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
print(f"🌲 Nombre de features : {len(features_selected)}")
print(f"📈 Courbes générées : Apprentissage + Validation + Distribution")
print(f"🏆 Meilleure feature : {feature_importance.iloc[0]['feature']}")

# 💡 Exemple d'utilisation
print("\n💡 EXEMPLE DE PRÉDICTION:")
print("Prédiction pour une commune type d'Île-de-France...")
example_prediction, example_proba = predire_orientation_ile_de_france(
    nb_inscrits=2500,
    nb_abstentions=750,
    pct_senior=22.0,
    pct_sans_activite=15.0,
    pct_etrangere=12.5,
    taux_chomage=8.5,
    nb_population_active=1800,
    population_totale=3200
)
print(f"🔮 Orientation prédite : {example_prediction}")
print("📊 Probabilités par classe :")
for classe, prob in sorted(example_proba.items(), key=lambda x: x[1], reverse=True):
    print(f"   {classe}: {prob:.3f} ({prob*100:.1f}%)")

print("\n🎯 Modèle prêt pour des prédictions sur l'Île-de-France ! 🚀")