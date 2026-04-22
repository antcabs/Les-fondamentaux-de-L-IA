# -*- coding: utf-8 -*-
"""
TP3 - Deep Learning : classification automatique de produits e-commerce
Matiere : Les fondamentaux de l'IA | Bachelor 3
Dataset  : Fashion-MNIST (Zalando Research, 2017)
Contexte : Data Scientist junior -- equipe Catalog Intelligence de Zalando
Objectif : Reduire le taux d'erreur de categorisation de 12% a < 5%
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================================
# ETAPE 1 -- Charger et explorer les donnees produits Zalando
# =======================================================================
print("=" * 65)
print("ETAPE 1 -- CHARGEMENT ET EXPLORATION DES DONNEES")
print("=" * 65)

import tensorflow as tf
from tensorflow import keras

# Chargement du dataset Zalando Fashion-MNIST (telechargement automatique ~30 Mo)
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Les 10 categories du catalogue Zalando
class_names = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]

print("=== EXPLORATION DU CATALOGUE ===")
print(f"Train : {X_train.shape[0]} images de {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Test  : {X_test.shape[0]} images")
print(f"Pixels : min={X_train.min()}, max={X_train.max()} (niveaux de gris 0-255)")
print(f"Categories : {len(class_names)}")

# Distribution des categories
print("\nDistribution des categories (train) :")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {class_names[u]:12s} : {c:>5d} articles ({c/len(y_train)*100:.1f}%)")

# Visualisation : exemples du catalogue
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i], fontsize=10)
    ax.axis('off')
plt.suptitle('Catalogue Zalando -- 1 exemple par categorie', fontsize=13)
plt.tight_layout()
plt.savefig('catalogue_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardee : catalogue_samples.png")

# =======================================================================
# ETAPE 2 -- Pretraitement des images
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 2 -- PRETRAITEMENT DES IMAGES")
print("=" * 65)

# Normalisation : [0, 255] -> [0.0, 1.0]
# Les reseaux de neurones convergent mieux avec des valeurs entre 0 et 1
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm  = X_test.astype('float32') / 255.0

print(f"Apres normalisation : min={X_train_norm.min():.1f}, max={X_train_norm.max():.1f}")

# Pour scikit-learn : aplatir chaque image 28x28 en vecteur de 784 valeurs
X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat  = X_test_norm.reshape(-1, 784)

print(f"Forme pour ML classique (aplatie)  : {X_train_flat.shape}")
print(f"Forme pour reseau dense (grille)   : {X_train_norm.shape}")

# =======================================================================
# ETAPE 3 -- Baseline : Random Forest (ML classique)
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 3 -- BASELINE : RANDOM FOREST (ML CLASSIQUE)")
print("=" * 65)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest sur les pixels aplatis (784 features = 784 pixels individuels)
print("Entrainement du Random Forest (100 arbres)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)

y_pred_rf = rf.predict(X_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest -- Accuracy : {acc_rf*100:.1f}%")
print(f"Random Forest -- Taux d'erreur : {(1-acc_rf)*100:.1f}%")

# =======================================================================
# ETAPE 4 -- Reseau de neurones dense (MLP)
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 4 -- RESEAU DE NEURONES DENSE (MLP)")
print("=" * 65)

# Architecture : Input(28x28) -> Flatten -> Dense(128) -> Dense(64) -> Dense(10)
model_dense = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),                        # 28x28 -> 784
    keras.layers.Dense(128, activation='relu'),    # Couche cachee 1
    keras.layers.Dense(64, activation='relu'),     # Couche cachee 2
    keras.layers.Dense(10, activation='softmax')   # 10 categories -> probabilites
])

model_dense.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Classification multi-classe avec labels entiers
    metrics=['accuracy']
)

model_dense.summary()

# Entrainement -- 15% du train reserve pour la validation
print("\nEntrainement du reseau dense (15 epoques)...")
history_dense = model_dense.fit(
    X_train_norm, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# =======================================================================
# ETAPE 5 -- Reseau convolutif (CNN)
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 5 -- RESEAU CONVOLUTIF (CNN)")
print("=" * 65)

# Ajouter la dimension canal (niveaux de gris = 1 canal, couleur RGB = 3 canaux)
X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)
print(f"Forme pour CNN : {X_train_cnn.shape}")  # (60000, 28, 28, 1)

model_cnn = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    # Bloc 1 : detection de motifs simples (contours, bords)
    keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 32 filtres 3x3
    keras.layers.MaxPooling2D((2, 2)),                   # Reduction spatiale

    # Bloc 2 : detection de motifs complexes (formes, structures)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filtres 3x3
    keras.layers.MaxPooling2D((2, 2)),                   # Reduction spatiale

    # Classification finale
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),                           # Regularisation anti-overfitting
    keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_cnn.summary()

print("\nEntrainement du CNN (10 epoques)...")
history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# =======================================================================
# ETAPE 6 -- Courbes d'apprentissage : diagnostic production
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 6 -- COURBES D'APPRENTISSAGE")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reseau dense
axes[0].plot(history_dense.history['accuracy'],     'b-', label='Train')
axes[0].plot(history_dense.history['val_accuracy'], 'r-', label='Validation')
axes[0].set_title('Reseau Dense (MLP)')
axes[0].set_xlabel('Epoque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# CNN
axes[1].plot(history_cnn.history['accuracy'],     'b-', label='Train')
axes[1].plot(history_cnn.history['val_accuracy'], 'r-', label='Validation')
axes[1].set_title('CNN')
axes[1].set_xlabel('Epoque')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'apprentissage -- Diagnostic overfitting", fontsize=13)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardee : learning_curves.png")

# Diagnostic overfitting
final_train_dense = history_dense.history['accuracy'][-1]
final_val_dense   = history_dense.history['val_accuracy'][-1]
final_train_cnn   = history_cnn.history['accuracy'][-1]
final_val_cnn     = history_cnn.history['val_accuracy'][-1]
gap_dense = final_train_dense - final_val_dense
gap_cnn   = final_train_cnn   - final_val_cnn

print(f"\nDiagnostic overfitting :")
print(f"  Reseau Dense -- Train: {final_train_dense*100:.1f}% | Val: {final_val_dense*100:.1f}% | Ecart: {gap_dense*100:.1f}%")
print(f"  CNN          -- Train: {final_train_cnn*100:.1f}% | Val: {final_val_cnn*100:.1f}% | Ecart: {gap_cnn*100:.1f}%")

# =======================================================================
# ETAPE 7 -- Comparaison des 3 approches et decision
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 7 -- COMPARAISON DES 3 APPROCHES ET DECISION")
print("=" * 65)

from sklearn.metrics import classification_report, confusion_matrix

# Evaluation sur le jeu de test (donnees jamais vues)
loss_dense, acc_dense = model_dense.evaluate(X_test_norm, y_test, verbose=0)
loss_cnn,   acc_cnn   = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)

# Tableau comparatif
print("=" * 55)
print(" COMPARAISON DES 3 APPROCHES -- COMITE TECHNIQUE")
print("=" * 55)
print(f" {'Modele':<20s} {'Accuracy':>10s} {'Taux erreur':>12s}")
print("-" * 55)
print(f" {'Random Forest':<20s} {acc_rf*100:>9.1f}% {(1-acc_rf)*100:>11.1f}%")
print(f" {'Reseau Dense':<20s} {acc_dense*100:>9.1f}% {(1-acc_dense)*100:>11.1f}%")
print(f" {'CNN':<20s} {acc_cnn*100:>9.1f}% {(1-acc_cnn)*100:>11.1f}%")
print("-" * 55)
print(f" Objectif business : taux d'erreur < 5.0%")
print("=" * 55)

# Predictions du CNN pour l'analyse detaillee
y_pred_cnn     = model_cnn.predict(X_test_cnn, verbose=0)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de confusion -- CNN (articles Zalando)')
plt.ylabel('Categorie reelle')
plt.xlabel('Categorie predite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure sauvegardee : confusion_matrix_cnn.png")

# Rapport detaille par categorie
print("\n=== RAPPORT PAR CATEGORIE -- CNN ===")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# =======================================================================
# ETAPE 8 -- Analyse des erreurs pour l'equipe produit
# =======================================================================
print("=" * 65)
print("ETAPE 8 -- ANALYSE DES ERREURS")
print("=" * 65)

# Articles mal classes par le CNN
errors = np.where(y_pred_classes != y_test)[0]
print(f"Erreurs : {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

# Afficher 10 erreurs types
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx], cmap='gray')
    confiance = y_pred_cnn[idx, y_pred_classes[idx]] * 100
    ax.set_title(
        f"Predit : {class_names[y_pred_classes[idx]]} ({confiance:.0f}%)\n"
        f"Reel : {class_names[y_test[idx]]}",
        fontsize=8,
        color='red'
    )
    ax.axis('off')
plt.suptitle('CNN -- Articles mal classes (analyse qualite)', fontsize=13)
plt.tight_layout()
plt.savefig('erreurs_cnn.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardee : erreurs_cnn.png")

# Analyse : quelles paires de categories posent le plus de problemes ?
print("\n=== TOP 5 CONFUSIONS LES PLUS FREQUENTES ===")
confusions = {}
for real, pred in zip(y_test[errors], y_pred_classes[errors]):
    pair = (class_names[real], class_names[pred])
    confusions[pair] = confusions.get(pair, 0) + 1

top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
for (real, pred), count in top_confusions:
    print(f"  {real:12s} -> classe comme {pred:12s} : {count} erreurs")

# Analyse de la confiance sur les erreurs
confidences_erreurs = [y_pred_cnn[idx, y_pred_classes[idx]] for idx in errors]
high_conf_errors = sum(1 for c in confidences_erreurs if c > 0.9)
print(f"\nErreurs avec confiance > 90% : {high_conf_errors} ({high_conf_errors/len(errors)*100:.1f}%)")

# =======================================================================
# ETAPE 9 -- Visualiser ce que le CNN a appris
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 9 -- VISUALISATION DE CE QUE LE CNN A APPRIS")
print("=" * 65)

# 9a -- Filtres appris par la premiere couche
print("9a -- Filtres de la 1re couche Conv2D")

# Extraire les 32 filtres 3x3 de la premiere couche convolutive
first_conv_layer = model_cnn.layers[0]
filters, biases = first_conv_layer.get_weights()
print(f"Filtres : {filters.shape}")  # (3, 3, 1, 32)

# Normaliser les filtres pour la visualisation
f_min, f_max = filters.min(), filters.max()
filters_norm = (filters - f_min) / (f_max - f_min)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters_norm[:, :, 0, i], cmap='gray')
    ax.set_title(f'F{i+1}', fontsize=7)
    ax.axis('off')
plt.suptitle('Filtres appris par le CNN -- 1re couche Conv2D (3x3)', fontsize=13)
plt.tight_layout()
plt.savefig('filtres_conv.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardee : filtres_conv.png")

# 9b -- Ce que le CNN voit sur un article
print("\n9b -- Activations du CNN sur un article")

# Modele intermediaire qui renvoie les activations de la 1re couche Conv2D
activation_model = keras.Model(
    inputs=model_cnn.input,
    outputs=model_cnn.layers[0].output
)

# Choisir un article (Sneaker, categorie 7)
sample_idx  = np.where(y_test == 7)[0][0]
sample      = X_test_cnn[sample_idx:sample_idx+1]
activations = activation_model.predict(sample, verbose=0)
print(f"Activations : {activations.shape}")  # (1, 26, 26, 32)

# Image originale + 8 feature maps
fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
axes[0].imshow(X_test[sample_idx], cmap='gray')
axes[0].set_title('Original', fontsize=9)
axes[0].axis('off')
for i in range(8):
    axes[i+1].imshow(activations[0, :, :, i], cmap='viridis')
    axes[i+1].set_title(f'Filtre {i+1}', fontsize=9)
    axes[i+1].axis('off')
plt.suptitle(f'Activations du CNN -- {class_names[y_test[sample_idx]]}', fontsize=12)
plt.tight_layout()
plt.savefig('activations_conv.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardee : activations_conv.png")

# =======================================================================
# ETAPE 10 -- Debrief : recommandation au comite technique
# =======================================================================
print("\n" + "=" * 65)
print("ETAPE 10 -- DEBRIEF : RECOMMANDATION AU COMITE TECHNIQUE")
print("=" * 65)

print(f"""
SYNTHESE POUR LE COMITE TECHNIQUE -- ZALANDO
Projet : Classification automatique du catalogue

INSIGHT 1 -- ML CLASSIQUE vs DEEP LEARNING
Random Forest  : {acc_rf*100:.1f}% accuracy ({(1-acc_rf)*100:.1f}% d'erreur)
Reseau Dense   : {acc_dense*100:.1f}% accuracy ({(1-acc_dense)*100:.1f}% d'erreur)
CNN            : {acc_cnn*100:.1f}% accuracy ({(1-acc_cnn)*100:.1f}% d'erreur)
Gain RF -> CNN : +{(acc_cnn-acc_rf)*100:.1f} points d'accuracy

Recommandation : Le surcoût GPU du CNN est justifie par le gain de performance.
Deployer le CNN en production.

INSIGHT 2 -- RESEAU DENSE vs CNN
Ecart Dense/CNN : {(acc_cnn-acc_dense)*100:.1f} points d'accuracy
Les convolutions capturent la structure spatiale (contours, formes)
que le MLP ignore en traitant les pixels comme independants.

INSIGHT 3 -- FIABILITE EN PRODUCTION
Erreurs avec confiance > 90% : {high_conf_errors} ({high_conf_errors/len(errors)*100:.1f}% des erreurs)
-> Recommandation : seuil de confiance > 85% pour publication automatique,
   validation humaine en dessous (environ 20% des articles).
""")

print("=" * 65)
print("FIN DU TP3 -- Livrables generes :")
print("  - catalogue_samples.png")
print("  - learning_curves.png")
print("  - confusion_matrix_cnn.png")
print("  - erreurs_cnn.png")
print("  - filtres_conv.png")
print("  - activations_conv.png")
print("=" * 65)
