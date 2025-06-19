# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.metrics import classification_report, confusion_matrix
# # from sklearn.utils.multiclass import unique_labels
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # === 1. Charger le dataset ===
# # fichier_2022 = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/jointure_elections_population_criminalite_emploi.csv"
# # df = pd.read_csv(fichier_2022, low_memory=False)

# # # === 2. Filtrer les départements 01 à 15 (région Auvergne-Rhône-Alpes) ===
# # df['ID Commune'] = df['ID Commune'].astype(str)
# # df = df[df['ID Commune'].str[:2].isin([f"{i:02}" for i in range(1, 16)])]

# # # === 3. Supprimer les colonnes non pertinentes ===
# # colonnes_a_supprimer = ['ID Commune', 'Année', 'Nom Complet Élu', 'Voix Élu', 'Sexe Élu']
# # df = df.drop(columns=[col for col in colonnes_a_supprimer if col in df.columns])

# # # === 4. Supprimer les lignes sans cibles ===
# # df.dropna(subset=['Parti Politique Élu', 'Score Orientation (0 à 4)'], inplace=True)

# # # === 5. Encoder la variable cible (Parti Politique Élu) ===
# # le_parti = LabelEncoder()
# # df['parti_encoded'] = le_parti.fit_transform(df['Parti Politique Élu'])

# # # === 6. Supprimer les classes trop rares (<2) pour éviter erreurs de stratification ===
# # counts = df['parti_encoded'].value_counts()
# # classes_valides = counts[counts >= 2].index
# # df = df[df['parti_encoded'].isin(classes_valides)]

# # # === 7. Définir X et y ===
# # X = df.select_dtypes(include=[np.number]).drop(columns=['Score Orientation (0 à 4)', 'parti_encoded'])
# # y = df['parti_encoded']

# # # === 8. Séparer en train/test avec stratification ===
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # # === 9. Entraîner le modèle Random Forest ===
# # clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# # clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)

# # # === 10. Afficher le rapport de classification (corrigé) ===
# # labels_utilisés = unique_labels(y_test, y_pred)
# # target_names_utilisés = le_parti.inverse_transform(labels_utilisés)

# # print("Classification Report - Parti Politique Élu :")
# # print(classification_report(y_test, y_pred, labels=labels_utilisés, target_names=target_names_utilisés))

# # # === 11. Afficher la matrice de confusion ===
# # cm = confusion_matrix(y_test, y_pred, labels=labels_utilisés)
# # plt.figure(figsize=(10, 6))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #             xticklabels=target_names_utilisés, yticklabels=target_names_utilisés)
# # plt.title("Confusion Matrix - Parti Politique Élu")
# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")
# # plt.tight_layout()
# # plt.show()

# # # # # # === 12. Afficher les importances des features ===
# # # # # importances = clf.feature_importances_
# # # # # feat_names = X.columns
# # # # # indices = np.argsort(importances)[::-1]

# # # # # plt.figure(figsize=(12, 6))
# # # # # sns.barplot(x=importances[indices][:15], y=feat_names[indices][:15])
# # # # # plt.title("Top 15 Feature Importances - Random Forest")
# # # # # plt.xlabel("Importance")
# # # # # plt.ylabel("Features")
# # # # # plt.tight_layout()
# # # # # plt.show()



# # # # import pandas as pd
# # # # from sklearn.ensemble import RandomForestClassifier
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.metrics import classification_report, accuracy_score
# # # # from sklearn.preprocessing import LabelEncoder
# # # # import matplotlib.pyplot as plt

# # # # # 📁 Chemin vers le fichier
# # # # fichier = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/jointure_elections_population_criminalite_emploi.csv"

# # # # # 📥 Charger les données avec traitement de colonnes mixtes
# # # # df = pd.read_csv(fichier, low_memory=False)

# # # # # ❌ Supprimer les colonnes non pertinentes
# # # # colonnes_a_supprimer = [
# # # #     "ID Commune", "Année", "Sexe Élu", "Nom Complet Élu", "Parti Politique Élu", "Score Orientation (0 à 4)"
# # # # ]
# # # # df = df.drop(columns=colonnes_a_supprimer)

# # # # # ✅ Supprimer les lignes avec valeurs manquantes
# # # # df = df.dropna(subset=["Orientation Politique"])  # Cible sans NaN
# # # # df = df.dropna()  # Supprimer toutes les lignes avec des NaN ailleurs

# # # # # 🔢 Encodage des variables catégorielles (hors cible)
# # # # encodeurs = {}
# # # # for col in df.select_dtypes(include=["object"]).columns:
# # # #     if col != "Orientation Politique":
# # # #         encoder = LabelEncoder()
# # # #         df[col] = encoder.fit_transform(df[col].astype(str))
# # # #         encodeurs[col] = encoder

# # # # # 🎯 Encodage de la cible
# # # # target_encoder = LabelEncoder()
# # # # df["Orientation Politique"] = target_encoder.fit_transform(df["Orientation Politique"])

# # # # # 📈 Séparer X et y
# # # # X = df.drop(columns=["Orientation Politique"])
# # # # y = df["Orientation Politique"]

# # # # # 🔀 Split train/test
# # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # # 🌳 Random Forest Classifier
# # # # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # # # model.fit(X_train, y_train)

# # # # # 🔍 Prédiction et évaluation
# # # # y_pred = model.predict(X_test)

# # # # print("✅ Rapport de classification :")
# # # # print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
# # # # print(f"🎯 Accuracy : {accuracy_score(y_test, y_pred):.2f}")

# # # # # 📊 Importance des variables
# # # # importances = model.feature_importances_
# # # # features = X.columns

# # # # plt.figure(figsize=(10, 6))
# # # # plt.barh(features, importances, color='orange')
# # # # plt.xlabel("Importance")
# # # # plt.title("🎯 Importance des variables - Random Forest")
# # # # plt.tight_layout()
# # # # plt.show()
# # # import pandas as pd
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import LabelEncoder
# # # from sklearn.metrics import classification_report, accuracy_score
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # # 📁 Chargement du dataset
# # # fichier = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/jointure_elections_population_criminalite_emploi.csv"
# # # df = pd.read_csv(fichier, low_memory=False)

# # # # ✅ Nettoyage : suppression des colonnes inutiles
# # # colonnes_a_supprimer = ["Nom Complet Élu", "Sexe Élu", "Année","% Voix/Ins Élu","% Voix/Exp Élu" ,"ID Commune","Parti Politique Élu","Score Orientation (0 à 4)"]
# # # df.drop(columns=colonnes_a_supprimer, inplace=True)

# # # # ✅ Suppression des lignes sans étiquette cible
# # # df.dropna(subset=["Orientation Politique"], inplace=True)

# # # # ✅ Encodage des colonnes catégorielles (sauf la cible)
# # # label_encoders = {}
# # # for col in df.select_dtypes(include="object").columns:
# # #     if col != "Orientation Politique":
# # #         le = LabelEncoder()
# # #         df[col] = le.fit_transform(df[col].astype(str))
# # #         label_encoders[col] = le

# # # # ✅ Encodage de la cible
# # # target_encoder = LabelEncoder()
# # # df["Orientation Politique"] = target_encoder.fit_transform(df["Orientation Politique"])

# # # # 🧮 AFFICHER LA RÉPARTITION DES CLASSES
# # # print("\n🔢 Répartition des classes dans Orientation Politique :")
# # # valeurs = df["Orientation Politique"].value_counts()
# # # for i, count in valeurs.items():
# # #     print(f"{target_encoder.inverse_transform([i])[0]} : {count}")

# # # # 🔀 Séparation des features / target
# # # X = df.drop(columns=["Orientation Politique"])
# # # y = df["Orientation Politique"]

# # # # 🔀 Split train/test
# # # X_train, X_test, y_train, y_test = train_test_split(
# # #     X, y, test_size=0.2, stratify=y, random_state=42
# # # )

# # # # 🌳 Modèle Random Forest (basique)
# # # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # # model.fit(X_train, y_train)

# # # # 📊 Prédiction et évaluation
# # # y_pred = model.predict(X_test)
# # # print("\n✅ Rapport de classification :")
# # # print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
# # # print(f"🎯 Accuracy : {accuracy_score(y_test, y_pred):.2f}")

# # # # 📉 Matrice de confusion
# # # plt.figure(figsize=(8, 6))
# # # cm = pd.crosstab(y_test, y_pred, rownames=["Réel"], colnames=["Prédit"])
# # # sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
# # #             xticklabels=target_encoder.classes_,
# # #             yticklabels=target_encoder.classes_)
# # # plt.title("Matrice de confusion - Orientation Politique")
# # # plt.tight_layout()
# # # plt.show()
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import LabelEncoder
# # # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import warnings

# # # # 📁 Chargement du dataset
# # # fichier = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/jointure_complete_validee.csv"
# # # df = pd.read_csv(fichier, low_memory=False)

# # # # ✅ Nettoyage : suppression des colonnes inutiles
# # # colonnes_a_supprimer = ["Nom Complet Élu", "Sexe Élu", "Année", "% Voix/Ins Élu", "% Voix/Exp Élu", "ID Commune", "Parti Politique Élu", "Score Orientation (0 à 4)"]
# # # df.drop(columns=colonnes_a_supprimer, inplace=True)

# # # # ✅ Suppression des lignes sans étiquette cible
# # # df.dropna(subset=["Orientation Politique"], inplace=True)

# # # # ✅ Encodage des colonnes catégorielles (sauf la cible)
# # # label_encoders = {}
# # # for col in df.select_dtypes(include="object").columns:
# # #     if col != "Orientation Politique":
# # #         le = LabelEncoder()
# # #         df[col] = le.fit_transform(df[col].astype(str))
# # #         label_encoders[col] = le

# # # # ✅ Encodage de la cible
# # # target_encoder = LabelEncoder()
# # # df["Orientation Politique"] = target_encoder.fit_transform(df["Orientation Politique"])

# # # # 🧮 Répartition des classes
# # # print("\n🔢 Répartition des classes dans Orientation Politique :")
# # # valeurs = df["Orientation Politique"].value_counts()
# # # for i, count in valeurs.items():
# # #     print(f"{target_encoder.inverse_transform([i])[0]} : {count}")

# # # # ⚠️ Optionnel : filtrer les classes avec trop peu d'exemples
# # # classes_rare = [i for i, count in valeurs.items() if count < 100]
# # # df = df[~df["Orientation Politique"].isin(classes_rare)]

# # # # 🔀 Features / target
# # # X = df.drop(columns=["Orientation Politique"])
# # # y = df["Orientation Politique"]

# # # # 🔀 Train/Test split
# # # X_train, X_test, y_train, y_test = train_test_split(
# # #     X, y, test_size=0.2, stratify=y, random_state=42
# # # )

# # # # 🌳 Modèle Random Forest
# # # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # # model.fit(X_train, y_train)

# # # # 📊 Prédiction
# # # y_pred = model.predict(X_test)

# # # # ✅ Rapport de classification
# # # print("\n✅ Rapport de classification :")
# # # warnings.filterwarnings("ignore")  # Pour éviter les warnings sklearn
# # # print(classification_report(y_test, y_pred, target_names=target_encoder.inverse_transform(np.unique(y))))

# # # # 🎯 Accuracy
# # # print(f"🎯 Accuracy : {accuracy_score(y_test, y_pred):.2f}")

# # # # 📉 Matrice de confusion
# # # cm = confusion_matrix(y_test, y_pred)
# # # plt.figure(figsize=(8, 6))
# # # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
# # #             xticklabels=target_encoder.inverse_transform(np.unique(y)),
# # #             yticklabels=target_encoder.inverse_transform(np.unique(y)))
# # # plt.title("Matrice de confusion - Orientation Politique")
# # # plt.xlabel("Prédit")
# # # plt.ylabel("Réel")
# # # plt.tight_layout()
# # # plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Chargement des données
# print("📊 Chargement des données...")
# file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
# data = pd.read_csv(file_path, sep=';')

# print(f"Dataset complet: {data.shape[0]} lignes")

# # Extraction du code région à partir de commune_id (2 premiers chiffres)
# data['code_region'] = data['commune_id'].astype(str).str[:2]

# # Trouver la région avec le plus de données
# region_counts = data['code_region'].value_counts()
# print("\n📍 Données par région:")
# print(region_counts.head(10))

# # Choisir la région avec le plus de données (généralement "75" pour Paris ou "69" pour Rhône)
# best_region = region_counts.index[0]
# print(f"\n🎯 Région sélectionnée: {best_region} avec {region_counts[best_region]} communes")

# # Filtrer sur cette région
# data_region = data[data['code_region'] == best_region].copy()
# print(f"✅ Dataset région: {data_region.shape[0]} lignes")

# # Vérifier les classes disponibles
# print(f"\nClasses d'orientation politique:")
# print(data_region['orientation_politique'].value_counts().sort_index())

# # Nettoyage des données
# print("\n🔧 Nettoyage des données...")
# # Supprimer les valeurs manquantes dans la cible
# data_region = data_region.dropna(subset=['orientation_politique'])

# # Sélection des features importantes (pas trop compliqué)
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

# print(f"📋 Features sélectionnées: {len(features_selected)}")
# for i, feature in enumerate(features_selected, 1):
#     print(f"  {i}. {feature}")

# # Préparation des données
# X = data_region[features_selected].copy()
# y = data_region['orientation_politique'].copy()

# # Remplir les valeurs manquantes par la médiane
# print("\n🔄 Remplissage des valeurs manquantes...")
# for col in X.columns:
#     missing = X[col].isnull().sum()
#     if missing > 0:
#         print(f"  {col}: {missing} valeurs manquantes")
#         X[col] = X[col].fillna(X[col].median())

# print(f"✅ Données finales: {X.shape[0]} lignes, {X.shape[1]} features")

# # Division train/test
# print("\n📦 Division des données...")
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"Train: {X_train.shape[0]} lignes")
# print(f"Test: {X_test.shape[0]} lignes")

# # Standardisation
# print("\n📏 Standardisation des données...")
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print("✅ Standardisation terminée")

# # Entraînement du Random Forest
# print("\n🌲 Entraînement Random Forest...")
# rf_model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=15,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     random_state=42,
#     n_jobs=-1
# )

# rf_model.fit(X_train_scaled, y_train)
# print("✅ Modèle entraîné")

# # Prédictions
# print("\n🎯 Prédictions...")
# y_pred = rf_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"🏆 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# # Rapport détaillé
# print("\n📊 RAPPORT DE CLASSIFICATION:")
# print("="*50)
# print(classification_report(y_test, y_pred))

# # Matrice de confusion
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Matrice de Confusion')
# plt.xlabel('Prédictions')
# plt.ylabel('Vraies Valeurs')
# plt.show()

# # Importance des features
# print("\n🔍 IMPORTANCE DES FEATURES:")
# print("="*40)
# feature_importance = pd.DataFrame({
#     'feature': features_selected,
#     'importance': rf_model.feature_importances_
# }).sort_values('importance', ascending=False)

# for i, (feature, importance) in enumerate(feature_importance.values, 1):
#     print(f"{i:2d}. {feature:<25} : {importance:.4f}")

# # Graphique importance
# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importance, x='importance', y='feature')
# plt.title('Importance des Features')
# plt.xlabel('Importance')
# plt.tight_layout()
# plt.show()

# # Fonction de prédiction simple
# def predire_orientation(nb_inscrits, nb_abstentions, pct_jeune, pct_senior, 
#                        pct_sans_activite, pct_etrangere,  
#                        taux_chomage, population_totale):
#     """
#     Prédire l'orientation politique d'une nouvelle commune
#     """
#     # Créer le tableau de données
#     nouvelle_donnee = np.array([[
#         nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
#         pct_sans_activite, pct_etrangere, 
#         taux_chomage, population_totale
#     ]])
    
#     # Standardiser
#     nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)
    
#     # Prédire
#     prediction = rf_model.predict(nouvelle_donnee_scaled)[0]
#     probabilite = rf_model.predict_proba(nouvelle_donnee_scaled)[0]
    
#     return prediction, probabilite

# # Test de la fonction
# print("\n🔮 TEST DE PRÉDICTION:")
# print("="*30)
# test_prediction, test_proba = predire_orientation(
#     nb_inscrits=500,
#     nb_abstentions=100,
#     pct_jeune=25.0,
#     pct_senior=20.0,
#     pct_sans_activite=30.0,
#     pct_etrangere=5.0,
#     taux_chomage=0.08,
#     population_totale=800
# )

# print(f"Orientation prédite: {test_prediction}")
# print("Probabilités:")
# classes = sorted(y.unique())
# for classe, prob in zip(classes, test_proba):
#     print(f"  Classe {classe}: {prob:.3f}")

# print(f"\n🎉 RÉSUMÉ FINAL:")
# print(f"📍 Région analysée: {best_region}")
# print(f"📊 Nombre de communes: {len(data_region)}")
# print(f"🎯 Accuracy obtenue: {accuracy*100:.2f}%")
# print(f"🌲 Features utilisées: {len(features_selected)}")



#************************************************************************************************************************

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
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

# # 🗺️ Filtrage sur la région Grand Est
# # Liste des codes départements Grand Est (10 départements)
# grand_est_depts = [
#     '08', # Ardennes
#     '10', # Aube
#     '51', # Marne
#     '52', # Haute-Marne
#     '54', # Meurthe-et-Moselle
#     '55', # Meuse
#     '57', # Moselle
#     '67', # Bas-Rhin
#     '68', # Haut-Rhin
#     '88'  # Vosges
# ]

# data['code_dept'] = data['commune_id'].astype(str).str[:2]
# data_region = data[data['code_dept'].isin(grand_est_depts)].copy()

# print(f"📍 Région sélectionnée : Grand Est ({', '.join(grand_est_depts)})")
# print(f"✅ Nombre de communes : {data_region.shape[0]}")

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
# for col in X.columns:
#     if X[col].isnull().sum() > 0:
#         X[col] = X[col].fillna(X[col].median())

# # 📦 Séparation train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # 📏 Standardisation
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 🌲 Entraînement Random Forest
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

# # 🔲 Matrice de confusion
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Matrice de confusion - Orientation Politique")
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

# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importance, x='importance', y='feature')
# plt.title("Importance des variables")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()

# # 🔮 Fonction de prédiction personnalisée adaptée
# def predire_orientation(nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
#                         pct_sans_activite, pct_etrangere, taille_commune,
#                         taux_chomage, population_totale):
#     new_data = np.array([[nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
#                           pct_sans_activite, pct_etrangere, taille_commune,
#                           taux_chomage, population_totale]])
#     new_data_scaled = scaler.transform(new_data)
#     prediction = rf_model.predict(new_data_scaled)[0]
#     proba = rf_model.predict_proba(new_data_scaled)[0]
#     return prediction, proba

# # 🧪 Test de la fonction
# print("\n🔮 TEST DE PRÉDICTION :")
# test_prediction, test_proba = predire_orientation(
#     nb_inscrits=900,
#     nb_abstentions=150,
#     pct_jeune=26.0,
#     pct_senior=18.0,
#     pct_sans_activite=22.0,
#     pct_etrangere=11.0,
#     taille_commune=3,
#     taux_chomage=0.10,
#     population_totale=1300
# )

# print(f"Orientation prédite : {test_prediction}")
# print("Probabilités associées :")
# for classe, proba in zip(sorted(y.unique()), test_proba):
#     print(f"  Classe {classe}: {proba:.3f}")

# # 📝 Résumé
# print("\n🎉 RÉSUMÉ FINAL :")
# print(f"📍 Région analysée : Grand Est ({', '.join(grand_est_depts)})")
# print(f"📊 Communes utilisées : {len(data_region)}")
# print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
# print(f"🌲 Nombre de features : {len(features_selected)}")



#/*******************************************************************************************************
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
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

# # 🗺️ Filtrage sur la région Grand Est
# grand_est_depts = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
# data['code_dept'] = data['commune_id'].astype(str).str[:2]
# data_region = data[data['code_dept'].isin(grand_est_depts)].copy()

# print(f"📍 Région sélectionnée : Grand Est ({', '.join(grand_est_depts)})")
# print(f"✅ Nombre de communes : {data_region.shape[0]}")

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
# for col in X.columns:
#     if X[col].isnull().sum() > 0:
#         X[col] = X[col].fillna(X[col].median())

# # 📦 Séparation train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # 📏 Standardisation
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 🌲 Entraînement Random Forest
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

# # 🔲 Matrice de confusion
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Matrice de confusion - Orientation Politique")
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

# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importance, x='importance', y='feature')
# plt.title("Importance des variables")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()

# # 🔮 Fonction de prédiction personnalisée (à utiliser plus tard si besoin)
# def predire_orientation(nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
#                         pct_sans_activite, pct_etrangere, taille_commune,
#                         taux_chomage, population_totale):
#     new_data = np.array([[nb_inscrits, nb_abstentions, pct_jeune, pct_senior,
#                           pct_sans_activite, pct_etrangere, taille_commune,
#                           taux_chomage, population_totale]])
#     new_data_scaled = scaler.transform(new_data)
#     prediction = rf_model.predict(new_data_scaled)[0]
#     proba = rf_model.predict_proba(new_data_scaled)[0]
#     return prediction, proba

# # 📝 Résumé
# print("\n🎉 RÉSUMÉ FINAL :")
# print(f"📍 Région analysée : Grand Est ({', '.join(grand_est_depts)})")
# print(f"📊 Communes utilisées : {len(data_region)}")
# print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
# print(f"🌲 Nombre de features : {len(features_selected)}")
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

# 🗺️ Filtrage sur la région Grand Est
grand_est_depts = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
data['code_dept'] = data['commune_id'].astype(str).str[:2]
data_region = data[data['code_dept'].isin(grand_est_depts)].copy()

print(f"📍 Région sélectionnée : Grand Est ({', '.join(grand_est_depts)})")
print(f"✅ Nombre de communes : {data_region.shape[0]}")

# 🎯 Vérification des classes
print("\nClasses d'orientation politique :")
print(data_region['orientation_politique'].value_counts().sort_index())

# 🔧 Nettoyage des données
data_region = data_region.dropna(subset=['orientation_politique'])

# 📋 Sélection des variables (features)
features_selected = [
    'annee',
    'nb_inscrits',
    'nb_abstentions',
    'pct_population_senior',
    'pct_population_sans_activite',
    'pct_population_etrangere',
    'taux_chomage_pct',
    'nb_population_active',
    'Population_Totale',
    'nb_crimes'
]
X = data_region[features_selected].copy()
y = data_region['orientation_politique'].copy()

# 🔄 Remplir les valeurs manquantes avec la médiane
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# 📦 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 📏 Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🌲 Entraînement Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
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

plt.figure(figsize=(12, 5))

# Courbe d'apprentissage (équivalent aux courbes de précision)
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_mean, label='Train', linewidth=2, marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation', linewidth=2, marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Taille du dataset d\'entraînement')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbes d\'apprentissage')
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

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, train_mean_val, label='Train', linewidth=2, marker='o')
plt.fill_between(n_estimators_range, train_mean_val - train_std_val, 
                 train_mean_val + train_std_val, alpha=0.1)
plt.plot(n_estimators_range, val_mean_val, label='Validation', linewidth=2, marker='s')
plt.fill_between(n_estimators_range, val_mean_val - val_std_val, 
                 val_mean_val + val_std_val, alpha=0.1)
plt.xlabel('Nombre d\'estimateurs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (n_estimators)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 🔲 Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Orientation Politique")
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

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title("Importance des variables")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 📊 Performance par classe
plt.figure(figsize=(10, 6))
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
sns.barplot(data=performance_df, x='Classe', y='Score', hue='Métrique')
plt.title('Performance par classe politique')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔮 Fonction de prédiction personnalisée
def predire_orientation(annee,nb_inscrits, nb_abstentions, pct_senior,
                        pct_sans_activite, pct_etrangere, taux_chomage,
                        nb_population_active, population_totale,nb_crimes):
    new_data = np.array([[annee,nb_inscrits, nb_abstentions, pct_senior,
                          pct_sans_activite, pct_etrangere, taux_chomage,
                          nb_population_active, population_totale,nb_crimes]])
    new_data_scaled = scaler.transform(new_data)
    prediction = rf_model.predict(new_data_scaled)[0]
    proba = rf_model.predict_proba(new_data_scaled)[0]
    return prediction, proba

# 📝 Résumé
print("\n🎉 RÉSUMÉ FINAL :")
print(f"📍 Région analysée : Grand Est ({', '.join(grand_est_depts)})")
print(f"📊 Communes utilisées : {len(data_region)}")
print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
print(f"🌲 Nombre de features : {len(features_selected)}")
print(f"📈 Courbes générées : Apprentissage + Validation")