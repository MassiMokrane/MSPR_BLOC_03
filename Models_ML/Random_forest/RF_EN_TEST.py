import pandas as pd
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
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

# 🗺️ Séparation France entière vs Bretagne
bretagne_depts = ['22', '29', '35', '56']
data['code_dept'] = data['commune_id'].astype(str).str[:2]

# 🇫🇷 Dataset d'entraînement : France entière SAUF Bretagne
data_train = data[~data['code_dept'].isin(bretagne_depts)].copy()
print(f"🇫🇷 Entraînement sur France entière (hors Bretagne) : {data_train.shape[0]} communes")

# 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Dataset de test : Bretagne uniquement
data_test = data[data['code_dept'].isin(bretagne_depts)].copy()
print(f"🏴󠁧󠁢󠁳󠁣󠁴󠁿 Test sur Bretagne ({', '.join(bretagne_depts)}) : {data_test.shape[0]} communes")

# 🎯 Vérification des classes
print("\n🇫🇷 Classes d'orientation politique (France entière - hors Bretagne) :")
print(data_train['orientation_politique'].value_counts().sort_index())

print("\n🏴󠁧󠁢󠁳󠁣󠁴󠁿 Classes d'orientation politique (Bretagne) :")
print(data_test['orientation_politique'].value_counts().sort_index())

# 🔧 Nettoyage des données
data_train = data_train.dropna(subset=['orientation_politique'])
data_test = data_test.dropna(subset=['orientation_politique'])

print(f"\nAprès nettoyage :")
print(f"🇫🇷 Entraînement : {data_train.shape[0]} communes")
print(f"🏴󠁧󠁢󠁳󠁣󠁴󠁿 Test : {data_test.shape[0]} communes")

# 📋 Sélection des variables (features)
features_selected = [
    'nb_inscrits',
    'nb_abstentions',
    'pct_population_jeune',
    'pct_population_sans_activite',
    'pct_population_etrangere',
    'taux_chomage_pct',
    'nb_population_active',
    'Population_Totale',
    'nb_crimes'
]

# Préparation des données d'entraînement et de test
X_train = data_train[features_selected].copy()
y_train = data_train['orientation_politique'].copy()
X_test = data_test[features_selected].copy()
y_test = data_test['orientation_politique'].copy()

print(f"\n📊 Dimensions finales :")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# 📏 Standardisation (fit sur train, transform sur train et test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🌲 Entraînement Random Forest avec class_weight='balanced'
print("\n🌲 Entraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Pour gérer le déséquilibre des classes
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# 🎯 Prédictions et évaluation
print("🎯 Évaluation du modèle sur la Bretagne...")
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🏆 Accuracy sur la Bretagne : {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📊 Rapport de classification (Test sur Bretagne) :")
print(classification_report(y_test, y_pred))

# 📈 COURBES D'APPRENTISSAGE (sur les données d'entraînement France)
print("\n📈 Calcul des courbes d'apprentissage (validation croisée sur France)...")
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
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

# Courbe d'apprentissage
plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, label='Train (France)', linewidth=2, marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation CV (France)', linewidth=2, marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Taille du dataset d\'entraînement')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbes d\'apprentissage\n(Entraînement sur France)')
plt.grid(True, alpha=0.3)

# 📊 COURBE DE VALIDATION (hyperparamètre n_estimators)
print("📊 Calcul de la courbe de validation...")
n_estimators_range = range(10, 201, 20)
train_scores_val, val_scores_val = validation_curve(
    RandomForestClassifier(class_weight='balanced', random_state=42),
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
plt.plot(n_estimators_range, train_mean_val, label='Train (France)', linewidth=2, marker='o')
plt.fill_between(n_estimators_range, train_mean_val - train_std_val, 
                 train_mean_val + train_std_val, alpha=0.1)
plt.plot(n_estimators_range, val_mean_val, label='Validation CV (France)', linewidth=2, marker='s')
plt.fill_between(n_estimators_range, val_mean_val - val_std_val, 
                 val_mean_val + val_std_val, alpha=0.1)
plt.xlabel('Nombre d\'estimateurs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (n_estimators)\n(Validation croisée sur France)')
plt.grid(True, alpha=0.3)

# 🔲 Matrice de confusion
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion\nTest sur Bretagne")
plt.xlabel("Prédit")
plt.ylabel("Réel")

# 📊 Performance par classe
plt.subplot(2, 2, 4)
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
plt.title('Performance par classe\n(Test sur Bretagne)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# 🔍 Importance des features
feature_importance = pd.DataFrame({
    'feature': features_selected,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n🔍 Importance des features (modèle entraîné sur France) :")
for i, row in feature_importance.iterrows():
    print(f"{row['feature']:<30} : {row['importance']:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title("Importance des variables\n(Modèle entraîné sur France entière)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 🔮 Fonction de prédiction personnalisée
def predire_orientation(nb_inscrits, nb_abstentions, pct_population_jeune,
                        pct_population_sans_activite, pct_population_etrangere, taux_chomage_pct,
                        nb_population_active, population_totale, nb_crimes):
    """Prédire l'orientation politique d'une commune avec le modèle entraîné sur France"""
    new_data = np.array([[nb_inscrits, nb_abstentions, pct_population_jeune,
                          pct_population_sans_activite, pct_population_etrangere, taux_chomage_pct,
                          nb_population_active, population_totale, nb_crimes]])
    new_data_scaled = scaler.transform(new_data)
    prediction = rf_model.predict(new_data_scaled)[0]
    proba = rf_model.predict_proba(new_data_scaled)[0]
    return prediction, proba

# 📊 Comparaison des distributions France vs Bretagne
print("\n📊 Comparaison des orientations politiques :")
print("🇫🇷 France (entraînement) :")
france_dist = data_train['orientation_politique'].value_counts(normalize=True).sort_index()
for classe, pct in france_dist.items():
    print(f"  {classe}: {pct:.3f} ({pct*100:.1f}%)")

print("\n🏴󠁧󠁢󠁳󠁣󠁴󠁿 Bretagne (test) :")
bretagne_dist = data_test['orientation_politique'].value_counts(normalize=True).sort_index()
for classe, pct in bretagne_dist.items():
    print(f"  {classe}: {pct:.3f} ({pct*100:.1f}%)")

# 📝 Résumé
print("\n🎉 RÉSUMÉ FINAL :")
print(f"🇫🇷 Entraînement : France entière (hors Bretagne) - {len(data_train)} communes")
print(f"🏴󠁧󠁢󠁳󠁣󠁴󠁿 Test : Bretagne ({', '.join(bretagne_depts)}) - {len(data_test)} communes")
print(f"🎯 Accuracy sur Bretagne : {accuracy*100:.2f}%")
print(f"🌲 Nombre de features : {len(features_selected)}")
print(f"📈 Courbes générées : Apprentissage + Validation (sur données France)")
print(f"⚖️ Class weight : balanced (pour gérer le déséquilibre)")
print("🔬 Approche : Généralisation d'un modèle national à une région spécifique")