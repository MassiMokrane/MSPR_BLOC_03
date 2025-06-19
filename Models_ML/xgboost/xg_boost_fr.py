import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 📊 Chargement des données
print("📊 Chargement des données...")
file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
data = pd.read_csv(file_path, sep=';')
print(f"Dataset complet: {data.shape[0]} lignes")

# 🇫🇷 Analyse sur la France entière
print(f"🇫🇷 Analyse sur la France entière")
data_region = data.copy()

# Affichage des départements présents
data_region['code_dept'] = data_region['commune_id'].astype(str).str[:2]
unique_depts = sorted(data_region['code_dept'].unique())
print(f"📍 Départements analysés : {len(unique_depts)} départements")
print(f"✅ Nombre total de communes : {data_region.shape[0]}")

# Affichage de la répartition par région (approximative basée sur les codes départements)
dept_counts = data_region['code_dept'].value_counts().sort_index()
print(f"\n📊 Top 10 des départements avec le plus de communes:")
print(dept_counts.head(10))

# 🎯 Vérification des classes
print("\n🎯 Classes d'orientation politique :")
orientation_counts = data_region['orientation_politique'].value_counts().sort_index()
print(orientation_counts)

# Affichage des pourcentages
print("\n📊 Répartition en pourcentages :")
orientation_pct = (orientation_counts / orientation_counts.sum() * 100).round(2)
for orientation, pct in orientation_pct.items():
    print(f"{orientation}: {pct}%")

# 🔧 Nettoyage des données
print(f"\n🔧 Nettoyage des données...")
print(f"Lignes avant nettoyage: {data_region.shape[0]}")
data_region = data_region.dropna(subset=['orientation_politique'])
print(f"Lignes après suppression des orientations manquantes: {data_region.shape[0]}")

# 🏷️ Encodage des labels pour XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data_region['orientation_politique'])
print(f"\n🏷️ Classes encodées : {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# 📋 Sélection des variables (features)
features_selected = [
    'nb_inscrits',
    'nb_abstentions',
    'pct_population_senior',
    'pct_population_etrangere',
    'taux_chomage_pct',
    'Population_Totale',
    'nb_crimes'
]

# Vérification des colonnes disponibles
print(f"\n📋 Vérification des features sélectionnées:")
for feature in features_selected:
    if feature in data_region.columns:
        missing_count = data_region[feature].isnull().sum()
        print(f"✅ {feature}: {missing_count} valeurs manquantes ({missing_count/len(data_region)*100:.1f}%)")
    else:
        print(f"❌ {feature}: Colonne non trouvée")

X = data_region[features_selected].copy()
y = y_encoded

print(f"\n📊 Dimensions finales:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {len(y)}")

# 🔄 Remplir les valeurs manquantes avec la médiane
print(f"\n🔄 Traitement des valeurs manquantes...")
for col in X.columns:
    missing_before = X[col].isnull().sum()
    if missing_before > 0:
        X[col] = X[col].fillna(X[col].median())
        print(f"  {col}: {missing_before} valeurs remplacées par la médiane ({X[col].median():.2f})")

# 📦 Séparation train/test
print(f"\n📦 Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train set: {X_train.shape[0]} communes")
print(f"Test set: {X_test.shape[0]} communes")

# Vérification de la répartition des classes dans les sets
print(f"\n📊 Répartition des classes dans le train set:")
train_classes = pd.Series(y_train).value_counts().sort_index()
for i, count in enumerate(train_classes):
    class_name = label_encoder.classes_[i]
    print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")

# 📏 Standardisation 
print(f"\n📏 Standardisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🚀 Entraînement XGBoost
print("\n🚀 Entraînement du modèle XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1,
    tree_method='hist',
    device='cpu'
)

# Entraînement avec données d'évaluation pour suivre les performances
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
print("⏳ Entraînement en cours...")
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=eval_set,
    verbose=False
)
print("✅ Modèle entraîné!")

# 🎯 Prédictions et évaluation
print("\n🎯 Évaluation du modèle...")
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🏆 Accuracy sur la France entière : {accuracy:.4f} ({accuracy*100:.2f}%)")

# Convertir les prédictions en labels originaux pour le rapport
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\n📊 Rapport de classification détaillé :")
print(classification_report(y_test_labels, y_pred_labels))

# 📈 COURBES D'APPRENTISSAGE
print("\n📈 Calcul des courbes d'apprentissage...")
try:
    # Réduction de la taille du sample pour les courbes d'apprentissage (performance)
    sample_size = min(5000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_sample = X_train_scaled[indices]
    y_sample = y_train[indices]
    
    train_sizes, train_scores, val_scores = learning_curve(
        xgb.XGBClassifier(
            n_estimators=100, 
            random_state=42, 
            eval_metric='mlogloss',
            tree_method='hist',
            device='cpu'
        ),
        X_sample, y_sample,
        cv=3,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
        random_state=42,
        scoring='accuracy'
    )
    
    # Calcul des moyennes et écarts-types
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    learning_curves_success = True
    print("✅ Courbes d'apprentissage calculées")
except Exception as e:
    print(f"⚠️ Erreur dans le calcul des courbes d'apprentissage: {e}")
    learning_curves_success = False

# Création des graphiques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analyse du modèle XGBoost - France entière', fontsize=16, fontweight='bold')

# Courbe d'apprentissage
if learning_curves_success:
    axes[0,0].plot(train_sizes, train_mean, label='Train', linewidth=2, marker='o', color='blue')
    axes[0,0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes[0,0].plot(train_sizes, val_mean, label='Validation', linewidth=2, marker='s', color='red')
    axes[0,0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    axes[0,0].set_xlabel('Taille du dataset d\'entraînement')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].set_title('Courbes d\'apprentissage')
    axes[0,0].grid(True, alpha=0.3)
else:
    axes[0,0].text(0.5, 0.5, 'Courbes d\'apprentissage\nnon disponibles', 
                   ha='center', va='center', transform=axes[0,0].transAxes)
    axes[0,0].set_title('Courbes d\'apprentissage')

# 📊 COURBE DE VALIDATION
print("📊 Calcul de la courbe de validation...")
try:
    # Sample plus petit pour la validation curve
    sample_size_val = min(2000, len(X_train_scaled))
    indices_val = np.random.choice(len(X_train_scaled), sample_size_val, replace=False)
    X_sample_val = X_train_scaled[indices_val]
    y_sample_val = y_train[indices_val]
    
    n_estimators_range = range(50, 201, 50)
    train_scores_val, val_scores_val = validation_curve(
        xgb.XGBClassifier(
            random_state=42, 
            eval_metric='mlogloss',
            tree_method='hist',
            device='cpu'
        ),
        X_sample_val, y_sample_val,
        param_name='n_estimators',
        param_range=n_estimators_range,
        cv=3,
        n_jobs=1,
        scoring='accuracy'
    )
    
    train_mean_val = np.mean(train_scores_val, axis=1)
    train_std_val = np.std(train_scores_val, axis=1)
    val_mean_val = np.mean(val_scores_val, axis=1)
    val_std_val = np.std(val_scores_val, axis=1)
    
    validation_curves_success = True
    print("✅ Courbe de validation calculée")
except Exception as e:
    print(f"⚠️ Erreur dans le calcul de la courbe de validation: {e}")
    validation_curves_success = False

if validation_curves_success:
    axes[0,1].plot(n_estimators_range, train_mean_val, label='Train', linewidth=2, marker='o', color='blue')
    axes[0,1].fill_between(n_estimators_range, train_mean_val - train_std_val, 
                           train_mean_val + train_std_val, alpha=0.1, color='blue')
    axes[0,1].plot(n_estimators_range, val_mean_val, label='Validation', linewidth=2, marker='s', color='red')
    axes[0,1].fill_between(n_estimators_range, val_mean_val - val_std_val, 
                           val_mean_val + val_std_val, alpha=0.1, color='red')
    axes[0,1].set_xlabel('Nombre d\'estimateurs')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].set_title('Courbe de validation (n_estimators)')
    axes[0,1].grid(True, alpha=0.3)
else:
    axes[0,1].text(0.5, 0.5, 'Courbe de validation\nnon disponible', 
                   ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Courbe de validation (n_estimators)')

# 📈 Courbe de perte d'entraînement XGBoost
try:
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    axes[1,0].plot(x_axis, results['validation_0']['mlogloss'], label='Train', color='blue')
    axes[1,0].plot(x_axis, results['validation_1']['mlogloss'], label='Test', color='red')
    axes[1,0].set_xlabel('Epochs')
    axes[1,0].set_ylabel('Log Loss')
    axes[1,0].set_title('Courbes de perte XGBoost')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
except Exception as e:
    axes[1,0].text(0.5, 0.5, f'Courbes de perte\nnon disponibles', 
                   ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Courbes de perte XGBoost')

# 🔍 Importance des features
feature_importance = pd.DataFrame({
    'feature': features_selected,
    'importance': xgb_model.feature_importances_
}).sort_values(by='importance', ascending=True)  # Ascending pour le barplot horizontal

axes[1,1].barh(range(len(feature_importance)), feature_importance['importance'], color='skyblue')
axes[1,1].set_yticks(range(len(feature_importance)))
axes[1,1].set_yticklabels(feature_importance['feature'])
axes[1,1].set_xlabel('Importance')
axes[1,1].set_title('Importance des variables')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Affichage détaillé de l'importance des features
print("\n🔍 Importance des features (XGBoost) :")
feature_importance_sorted = feature_importance.sort_values(by='importance', ascending=False)
for i, row in feature_importance_sorted.iterrows():
    print(f"{row['feature']:<30} : {row['importance']:.4f}")

# 🔲 Matrice de confusion
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Nombre de communes'})
plt.title("Matrice de confusion - Orientation Politique France entière (XGBoost)", fontsize=14)
plt.xlabel("Orientation prédite")
plt.ylabel("Orientation réelle")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 📊 Performance par classe
plt.figure(figsize=(12, 8))
report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)
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
plt.title('Performance par classe politique - France entière (XGBoost)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(title='Métriques')
plt.tight_layout()
plt.show()

# 📊 Analyse par département (top 20)
print("\n📊 Analyse de performance par département (top 20 en nombre de communes)...")
data_test = data_region.iloc[X_test.index].copy()
data_test['prediction'] = y_pred_labels
data_test['actual'] = y_test_labels
data_test['correct'] = data_test['prediction'] == data_test['actual']

dept_performance = data_test.groupby('code_dept').agg({
    'correct': ['count', 'sum', 'mean'],
    'commune_id': 'count'
}).round(3)

dept_performance.columns = ['total_predictions', 'correct_predictions', 'accuracy', 'total_communes']
dept_performance = dept_performance[dept_performance['total_predictions'] >= 10]  # Minimum 10 communes
dept_performance = dept_performance.sort_values('total_communes', ascending=False).head(20)

print("Top 20 départements par nombre de communes testées:")
print(dept_performance)

# 🔮 Fonction de prédiction personnalisée
def predire_orientation(nb_inscrits, nb_abstentions, pct_population_senior,
                       pct_population_sans_activite, 
                       pct_population_etrangere, taux_chomage_pct,
                       nb_population_active, population_totale, nb_crimes):
    """
    Prédire l'orientation politique d'une commune française
    
    Paramètres:
    - nb_inscrits: Nombre d'inscrits sur les listes électorales
    - nb_abstentions: Nombre d'abstentions
    - pct_population_senior: Pourcentage de population senior
    - pct_population_sans_activite: Pourcentage de population sans activité
    - pct_population_etrangere: Pourcentage de population étrangère  
    - taux_chomage_pct: Taux de chômage en pourcentage
    - nb_population_active: Nombre de population active
    - population_totale: Population totale
    - nb_crimes: Nombre de crimes
    
    Retourne:
    - prediction: Orientation politique prédite
    - proba_dict: Dictionnaire des probabilités par classe
    """
    new_data = np.array([[nb_inscrits, nb_abstentions, pct_population_senior,
                        pct_population_sans_activite, 
                         pct_population_etrangere, taux_chomage_pct,
                         nb_population_active, population_totale, nb_crimes]])
    new_data_scaled = scaler.transform(new_data)
    prediction_encoded = xgb_model.predict(new_data_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    proba = xgb_model.predict_proba(new_data_scaled)[0]
    
    # Créer un dictionnaire avec les probabilités par classe
    proba_dict = {label_encoder.classes_[i]: proba[i] for i in range(len(proba))}
    
    return prediction, proba_dict

# Exemple d'utilisation de la fonction de prédiction
print("\n🔮 Exemple de prédiction pour une commune type:")
exemple_prediction, exemple_probas = predire_orientation(
    nb_inscrits=1000,
    nb_abstentions=300,
    pct_population_senior=25.0,
    pct_population_sans_activite=15.0,
    pct_population_etrangere=5.0,
    taux_chomage_pct=8.0,
    nb_population_active=800,
    population_totale=1500,
    nb_crimes=10
)

print(f"Prédiction: {exemple_prediction}")
print("Probabilités par classe:")
for classe, proba in sorted(exemple_probas.items(), key=lambda x: x[1], reverse=True):
    print(f"  {classe}: {proba:.3f} ({proba*100:.1f}%)")

# 📝 Résumé final
print("\n" + "="*80)
print("🎉 RÉSUMÉ FINAL - ANALYSE FRANCE ENTIÈRE")
print("="*80)
print(f"🇫🇷 Périmètre d'analyse : France entière")
print(f"📊 Communes analysées : {len(data_region):,}")
print(f"🏛️ Départements couverts : {len(unique_depts)}")
print(f"🎯 Accuracy obtenue : {accuracy*100:.2f}%")
print(f"🚀 Modèle utilisé : XGBoost (CPU, {xgb_model.n_estimators} estimateurs)")
print(f"🌲 Nombre de features : {len(features_selected)}")
print(f"📈 Classes prédites : {len(label_encoder.classes_)} orientations politiques")
print(f"💾 Modèle prêt pour les prédictions sur toute commune française")
print(f"⚡ Set d'entraînement : {len(X_train):,} communes")
print(f"🧪 Set de test : {len(X_test):,} communes")
print("="*80)

# Statistiques détaillées par classe
print(f"\n📊 RÉPARTITION DES ORIENTATIONS POLITIQUES (France entière):")
print("-" * 60)
for i, classe in enumerate(label_encoder.classes_):
    count = (data_region['orientation_politique'] == classe).sum()
    percentage = count / len(data_region) * 100
    print(f"{classe:<25} : {count:>6} communes ({percentage:>5.1f}%)")
print("-" * 60)
print(f"{'TOTAL':<25} : {len(data_region):>6} communes (100.0%)")