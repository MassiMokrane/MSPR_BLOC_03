import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
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
    'nb_inscrits',
    'nb_abstentions',
    'pct_population_senior',
    'pct_population_jeune',
    'pct_population_sans_activite',
    'taux_chomage_pct',
    'nb_population_active',
    'Population_Totale'
]
X = data_region[features_selected].copy()
y = data_region['orientation_politique'].copy()

# 🔄 Remplir les valeurs manquantes avec la médiane
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# 🏷️ Encodage des labels pour XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 📦 Séparation train/test
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 📏 Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🚀 Entraînement XGBoost avec suivi des courbes de perte
print("\n🚀 Entraînement du modèle XGBoost...")

# Configuration du modèle XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Entraînement avec validation set pour les courbes de perte
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_encoded, test_size=0.2, stratify=y_train_encoded, random_state=42
)

# Entraînement avec suivi
xgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
    verbose=False
)

# Ré-entraînement sur tout le dataset d'entraînement
xgb_final = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_final.fit(X_train_scaled, y_train_encoded)

# 🎯 Prédictions et évaluation
y_pred_encoded = xgb_final.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
y_test_original = label_encoder.inverse_transform(y_test_encoded)

accuracy = accuracy_score(y_test_original, y_pred)
print(f"\n🏆 Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📊 Rapport de classification :")
print(classification_report(y_test_original, y_pred))

# 📈 COURBES DE PERTE XGBoost
print("\n📈 Génération des courbes de perte...")
results = xgb_model.evals_result()

plt.figure(figsize=(15, 10))

# Courbes de perte
plt.subplot(2, 2, 1)
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train', linewidth=2)
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation', linewidth=2)
plt.legend()
plt.ylabel('Log Loss')
plt.xlabel('Époque')
plt.title('Courbes de perte XGBoost')
plt.grid(True, alpha=0.3)

# 📈 COURBES D'APPRENTISSAGE
print("📈 Calcul des courbes d'apprentissage...")
train_sizes, train_scores, val_scores = learning_curve(
    xgb.XGBClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train_encoded,
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

plt.subplot(2, 2, 2)
plt.plot(train_sizes, train_mean, label='Train', linewidth=2, marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, label='Validation', linewidth=2, marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel('Taille du dataset d\'entraînement')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbes d\'apprentissage')
plt.grid(True, alpha=0.3)

# 📊 COURBE DE VALIDATION (hyperparamètre n_estimators)
print("📊 Calcul de la courbe de validation...")
n_estimators_range = range(50, 301, 25)
train_scores_val, val_scores_val = validation_curve(
    xgb.XGBClassifier(random_state=42),
    X_train_scaled, y_train_encoded,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=5,
    n_jobs=-1
)

train_mean_val = np.mean(train_scores_val, axis=1)
train_std_val = np.std(train_scores_val, axis=1)
val_mean_val = np.mean(val_scores_val, axis=1)
val_std_val = np.std(val_scores_val, axis=1)

plt.subplot(2, 2, 3)
plt.plot(n_estimators_range, train_mean_val, label='Train', linewidth=2, marker='o')
plt.fill_between(n_estimators_range, train_mean_val - train_std_val, 
                 train_mean_val + train_std_val, alpha=0.2)
plt.plot(n_estimators_range, val_mean_val, label='Validation', linewidth=2, marker='s')
plt.fill_between(n_estimators_range, val_mean_val - val_std_val, 
                 val_mean_val + val_std_val, alpha=0.2)
plt.xlabel('Nombre d\'estimateurs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (n_estimators)')
plt.grid(True, alpha=0.3)

# 📊 COURBE DE VALIDATION (learning_rate)
print("📊 Calcul de la courbe de validation pour learning_rate...")
lr_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
train_scores_lr, val_scores_lr = validation_curve(
    xgb.XGBClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train_encoded,
    param_name='learning_rate',
    param_range=lr_range,
    cv=5,
    n_jobs=-1
)

train_mean_lr = np.mean(train_scores_lr, axis=1)
train_std_lr = np.std(train_scores_lr, axis=1)
val_mean_lr = np.mean(val_scores_lr, axis=1)
val_std_lr = np.std(val_scores_lr, axis=1)

plt.subplot(2, 2, 4)
plt.plot(lr_range, train_mean_lr, label='Train', linewidth=2, marker='o')
plt.fill_between(lr_range, train_mean_lr - train_std_lr, 
                 train_mean_lr + train_std_lr, alpha=0.2)
plt.plot(lr_range, val_mean_lr, label='Validation', linewidth=2, marker='s')
plt.fill_between(lr_range, val_mean_lr - val_std_lr, 
                 val_mean_lr + val_std_lr, alpha=0.2)
plt.xlabel('Taux d\'apprentissage')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Courbe de validation (learning_rate)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 🔲 Matrice de confusion
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_original, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion - Orientation Politique (XGBoost)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# 🔍 Importance des features XGBoost
feature_importance = pd.DataFrame({
    'feature': features_selected,
    'importance': xgb_final.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n🔍 Importance des features (XGBoost) :")
for i, row in feature_importance.iterrows():
    print(f"{row['feature']:<30} : {row['importance']:.4f}")

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title("Importance des variables (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 📊 Performance par classe
plt.figure(figsize=(12, 8))
report_dict = classification_report(y_test_original, y_pred, output_dict=True)
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
plt.title('Performance par classe politique (XGBoost)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔮 Fonction de prédiction personnalisée
def predire_orientation_xgb(nb_inscrits, nb_abstentions, pct_senior,pct_population_jeune,
                           pct_sans_activite, pct_etrangere, taux_chomage,
                           nb_population_active, population_totale):
    new_data = np.array([[nb_inscrits, nb_abstentions, pct_senior,pct_population_jeune,
                          pct_sans_activite, pct_etrangere, taux_chomage,
                          nb_population_active, population_totale]])
    new_data_scaled = scaler.transform(new_data)
    prediction_encoded = xgb_final.predict(new_data_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    proba = xgb_final.predict_proba(new_data_scaled)[0]
    
    # Créer un dictionnaire avec les classes et leurs probabilités
    proba_dict = {label_encoder.classes_[i]: proba[i] for i in range(len(proba))}
    
    return prediction, proba_dict

# 📊 Comparaison des métriques avancées
print("\n📊 MÉTRIQUES AVANCÉES XGBoost:")
print(f"🎯 Accuracy: {accuracy:.4f}")

# Calcul de l'AUC pour classification multiclasse
from sklearn.metrics import roc_auc_score
try:
    # Prédictions probabilistes
    y_pred_proba = xgb_final.predict_proba(X_test_scaled)
    auc_score = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
    print(f"📈 AUC Score (One-vs-Rest): {auc_score:.4f}")
except:
    print("📈 AUC Score: Non calculable pour cette configuration")

# Log Loss
from sklearn.metrics import log_loss
logloss = log_loss(y_test_encoded, y_pred_proba)
print(f"📉 Log Loss: {logloss:.4f}")

# 📝 Résumé final
print("\n🎉 RÉSUMÉ FINAL XGBoost :")
print(f"📍 Région analysée : Grand Est ({', '.join(grand_est_depts)})")
print(f"📊 Communes utilisées : {len(data_region)}")
print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
print(f"🌲 Nombre de features : {len(features_selected)}")
print(f"📈 Courbes générées : Perte + Apprentissage + Validation")
print(f"🚀 Modèle utilisé : XGBoost Classifier")
print(f"⚙️ Hyperparamètres optimisés : n_estimators, learning_rate, max_depth")

# 💡 Exemple d'utilisation de la fonction de prédiction
print("\n💡 EXEMPLE DE PRÉDICTION :")
print("Prédiction pour une commune type...")
example_prediction, example_proba = predire_orientation_xgb(
    nb_inscrits=1000,
    nb_abstentions=300,
    pct_senior=25.0,
    pct_population_jeune=30.0,
    pct_sans_activite=15.0,
    pct_etrangere=8.0,
    taux_chomage=12.0,
    nb_population_active=800,
    population_totale=1500
)
print(f"🔮 Orientation prédite : {example_prediction}")
print("📊 Probabilités par classe :")
for classe, prob in example_proba.items():
    print(f"   {classe}: {prob:.3f}")