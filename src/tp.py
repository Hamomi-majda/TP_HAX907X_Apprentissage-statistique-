
# Import des bibliothèques nécessaires
from sklearn import datasets  # Pour charger le dataset Iris
from sklearn.model_selection import train_test_split  # Pour diviser les données en train/test
from sklearn.svm import SVC  # Pour créer un modèle SVM
from sklearn.metrics import accuracy_score  # Pour évaluer la précision du modèle
import seaborn as sns  # Pour la visualisation (si nécessaire)
import matplotlib.pyplot as plt  # Pour tracer des graphiques

# 1. Charger le dataset Iris
iris = datasets.load_iris()
X = iris.data  # Données d'entrée (caractéristiques)
y = iris.target  # Étiquettes (classes)

# 2. Filtrer pour garder uniquement les classes 1 et 2 et les deux premières variables
X = X[y != 0, :2]  # On garde les deux premières variables seulement
y = y[y != 0]  # On filtre pour les classes 1 et 2

# 3. Diviser les données en ensembles d'entraînement et de test (50% pour chacun)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 4. Créer un modèle SVM avec un noyau linéaire
svm_linear = SVC(kernel='linear')

# 5. Entraîner le modèle sur les données d'entraînement
svm_linear.fit(X_train, y_train)

# 6. Prédire les étiquettes des données de test
y_pred = svm_linear.predict(X_test)

# 7. Évaluer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du SVM avec noyau linéaire: {accuracy:.2f}")

# Optionnel : Visualisation de la séparation des classes
plt.figure()
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, style=y_pred)
plt.title("Séparation des classes avec SVM linéaire")
plt.show()

