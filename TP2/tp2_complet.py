"""
TP2 — Cas C — Cybersécurité (Détection d'intrusions)
Matière : Les fondamentaux de l'IA | Bachelor 3
Dataset  : NSL-KDD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Cadrage du projet
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ÉTAPE 1 — CADRAGE DU PROJET")
print("=" * 65)
print("""
Problème métier : Identifier si une connexion réseau est une attaque
Variable cible  : label_binary — 0 = normal, 1 = attaque
Type de tâche   : Classification binaire
Métrique ML     : F1-macro (dataset déséquilibré, les deux erreurs coûtent)
Niveau AI Act   : HAUT RISQUE — système de sécurité critique
Erreur la plus coûteuse : Faux Négatif (attaque non détectée)
""")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Préparation des données
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ÉTAPE 2 — PRÉPARATION DES DONNÉES")
print("=" * 65)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

print("Chargement du dataset NSL-KDD depuis GitHub...")
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"
df = pd.read_csv(url, header=None, names=columns)
df['label_binary'] = (df['label'] != 'normal').astype(int)

print(f"Forme du dataset : {df.shape}")
print(f"\nDistribution des classes :")
print(f"  Normal  : {(df['label_binary']==0).sum()} ({(df['label_binary']==0).mean()*100:.1f}%)")
print(f"  Attaque : {(df['label_binary']==1).sum()} ({(df['label_binary']==1).mean()*100:.1f}%)")
print(f"\nTop 10 types d'attaques :")
print(df[df['label'] != 'normal']['label'].value_counts().head(10))

# ── Graphique distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df['label_binary'].value_counts()
axes[0].bar(['Normal (0)', 'Attaque (1)'], counts.values, color=['steelblue', 'tomato'])
axes[0].set_title('Distribution des classes (binaire)')
axes[0].set_ylabel('Nombre de connexions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
top_attacks = df[df['label'] != 'normal']['label'].value_counts().head(8)
axes[1].barh(top_attacks.index, top_attacks.values, color='tomato')
axes[1].set_title("Top 8 types d'attaques")
axes[1].set_xlabel("Nombre d'occurrences")
plt.tight_layout()
plt.savefig('distribution_classes.png', dpi=100)
plt.show()
print("Graphique sauvegardé : distribution_classes.png")

# ── Encodage + split + normalisation ──────────────────────────────────
print("\nEncodage des variables catégorielles :")
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"  {col} : {le.classes_}")

X = df.drop(['label', 'difficulty', 'label_binary'], axis=1)
y = df['label_binary']
feature_names = list(X.columns)

print(f"\nNombre de features : {len(feature_names)}")
print(f"Aucune valeur manquante : {X.isnull().sum().sum() == 0}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain : {len(X_train)} exemples | Test : {len(X_test)} exemples")
print(f"Taux d'attaque train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Modélisation : 3 modèles à comparer
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 3 — MODÉLISATION : 3 MODÈLES")
print("=" * 65)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(
                               n_estimators=100, random_state=42,
                               eval_metric='logloss', verbosity=0, n_jobs=-1
                           ),
}

resultats = {}
for nom, modele in modeles.items():
    print(f"Entraînement : {nom}...", end=" ", flush=True)
    modele.fit(X_train_sc, y_train)
    pred   = modele.predict(X_test_sc)
    acc    = accuracy_score(y_test, pred)
    f1_mac = f1_score(y_test, pred, average='macro')
    f1_wei = f1_score(y_test, pred, average='weighted')
    resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
    print(f"OK → Accuracy : {acc*100:.1f}% | F1-macro : {f1_mac:.3f} | F1-weighted : {f1_wei:.3f}")

df_resultats = pd.DataFrame(resultats).T
print("\n=== TABLEAU COMPARATIF ===")
print(df_resultats.round(3))

# ── Graphique comparaison ──────────────────────────────────────────────
ax = df_resultats[['accuracy', 'f1_macro']].plot(
    kind='bar', figsize=(9, 4), color=['steelblue', 'tomato']
)
plt.title('Comparaison des 3 modèles — Cas C Cybersécurité')
plt.ylabel('Score')
plt.xticks(rotation=20, ha='right')
plt.ylim(0.5, 1.02)
plt.grid(axis='y', alpha=0.5)
plt.legend(['Accuracy', 'F1-macro'])
for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.3f}',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('comparaison_modeles.png', dpi=100)
plt.show()
print("Graphique sauvegardé : comparaison_modeles.png")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Évaluation approfondie
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 4 — ÉVALUATION APPROFONDIE")
print("=" * 65)

from sklearn.metrics import confusion_matrix, classification_report

NOM_MEILLEUR = max(resultats, key=lambda x: resultats[x]['f1_macro'])
print(f"Meilleur modèle identifié : {NOM_MEILLEUR}")

meilleur = modeles[NOM_MEILLEUR]
y_pred   = meilleur.predict(X_test_sc)

print(f"\n=== RAPPORT DÉTAILLÉ — {NOM_MEILLEUR} ===")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attaque (1)']))

# ── Matrice de confusion ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("=== MATRICE DE CONFUSION ===")
print(f"  Vrais Négatifs  (TN) : {tn}  — connexions normales correctement identifiées")
print(f"  Faux Positifs   (FP) : {fp}  — connexions normales classées comme attaque (fausse alarme)")
print(f"  Faux Négatifs   (FN) : {fn}  — ATTAQUES NON DÉTECTÉES (le plus dangereux !)")
print(f"  Vrais Positifs  (TP) : {tp}  — attaques correctement détectées")

plt.figure(figsize=(7, 5))
labels = [[f'TN\n{tn}\n(Normal→Normal)', f'FP\n{fp}\n(Normal→Attaque)'],
          [f'FN\n{fn}\n(Attaque→Normal)', f'TP\n{tp}\n(Attaque→Attaque)']]
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=['Prédit Normal', 'Prédit Attaque'],
            yticklabels=['Réel Normal', 'Réel Attaque'],
            linewidths=2)
plt.title(f"Matrice de confusion — {NOM_MEILLEUR}\nCas C : Détection d'intrusions")
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()
print("Graphique sauvegardé : confusion_matrix.png")

# ── Évolution Random Forest selon n_estimators ─────────────────────────
n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []
print("\nÉvolution Random Forest selon n_estimators :")
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train_sc, y_train)
    score = f1_score(y_test, rf_temp.predict(X_test_sc), average='macro')
    scores_rf.append(score)
    print(f"  n={n:3d} arbres → F1-macro : {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o', linewidth=2, markersize=8)
for x, y_val in zip(n_estimators_range, scores_rf):
    plt.annotate(f'{y_val:.3f}', (x, y_val), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances — Random Forest (NSL-KDD)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('rf_evolution.png', dpi=100)
plt.show()
print("Graphique sauvegardé : rf_evolution.png")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Explicabilité SHAP
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 5 — EXPLICABILITÉ SHAP")
print("=" * 65)

import shap

rf_model = modeles["Random Forest"]
X_sample = np.array(X_test_sc[:300])

print("Calcul des valeurs SHAP (peut prendre ~30 secondes)...")
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Compatibilité SHAP < 0.42 (liste) et SHAP >= 0.42 (tableau 3D)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

print(f"Shape des SHAP values : {sv.shape}")
print("Calcul terminé !")

# ── Summary plot ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_sample, feature_names=feature_names, max_display=15, show=False)
plt.title("SHAP — Top 15 variables les plus influentes\n(classe positive = Attaque)")
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
plt.show()
print("Graphique sauvegardé : shap_summary.png")

# ── Bar chart importance SHAP ──────────────────────────────────────────
mean_shap = np.abs(sv).mean(axis=0)
top10_idx  = np.argsort(mean_shap)[::-1][:10]

print("\n=== TOP 10 VARIABLES LES PLUS INFLUENTES (SHAP) ===")
for i, idx in enumerate(top10_idx):
    print(f"  {i+1:2d}. {feature_names[idx]:35s} → SHAP moyen : {mean_shap[idx]:.4f}")

plt.figure(figsize=(9, 5))
plt.barh([feature_names[i] for i in top10_idx[::-1]],
         mean_shap[top10_idx[::-1]], color='steelblue')
plt.xlabel('Importance SHAP moyenne |valeur|')
plt.title("Top 10 features — Importance SHAP (Détection d'intrusions)")
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=100)
plt.show()
print("Graphique sauvegardé : shap_bar.png")

# ═══════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Conformité AI Act
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE 6 — CONFORMITÉ AI ACT")
print("=" * 65)
print("""
Niveau de risque  : HAUT RISQUE (Annexe III — infrastructures critiques)
Base légale RGPD  : Intérêt légitime (Art. 6.1.f) — logs réseau sans données perso
DPIA requis       : Oui si déployé sur logs réels avec adresses IP
Explicabilité     : Oui — SHAP pour chaque décision de blocage
Supervision humaine : Oui — analyste SOC valide les blocages définitifs
Audit             : MLflow versioning + logs horodatés + drift régulier
Organisme contrôle: ANSSI (cybersécurité) + CNIL (si données perso)
""")

# ═══════════════════════════════════════════════════════════════════════
# SLIDE DE SYNTHÈSE
# ═══════════════════════════════════════════════════════════════════════
metriques = resultats[NOM_MEILLEUR]
top3 = [feature_names[i] for i in top10_idx[:3]]

slide = f"""
╔══════════════════════════════════════════════════════════════════╗
║  CAS : C — Cybersécurité      MODÈLE RETENU : {NOM_MEILLEUR:<17}║
╠══════════════════════════════════════════════════════════════════╣
║  MÉTRIQUES FINALES                                               ║
║  Accuracy : {metriques['accuracy']*100:.1f}%   F1-macro : {metriques['f1_macro']:.3f}   F1-weighted : {metriques['f1_weighted']:.3f}  ║
╠══════════════════════════════════════════════════════════════════╣
║  TOP 3 VARIABLES (SHAP)                                          ║
║  1. {top3[0]:<20s}  2. {top3[1]:<20s}                ║
║  3. {top3[2]:<20s}                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  AI ACT : Niveau HAUT RISQUE                                     ║
║  Système de sécurité critique — supervision humaine obligatoire  ║
╠══════════════════════════════════════════════════════════════════╣
║  LIMITE PRINCIPALE : Dataset académique (NSL-KDD 2009),          ║
║  nécessite réentraînement sur des attaques récentes (2020+)      ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(slide)

# ── Récapitulatif fichiers générés ─────────────────────────────────────
fichiers = ['distribution_classes.png', 'comparaison_modeles.png',
            'confusion_matrix.png', 'rf_evolution.png',
            'shap_summary.png', 'shap_bar.png']
print("=== FICHIERS GÉNÉRÉS ===")
for f in fichiers:
    print(f"  {'[OK]' if os.path.exists(f) else '[MANQUANT]'} {f}")

print("\nTP2 — Cas C terminé !")
