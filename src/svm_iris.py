# %%
###############################################################################
#               Iris Dataset
###############################################################################

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Charger les données Iris
iris = datasets.load_iris()
X = iris.data
scaler = StandardScaler()  # Normalisation des données
X = scaler.fit_transform(X)
y = iris.target

# Filtrer pour garder uniquement les classes 1 et 2 et les deux premières variables
X = X[y != 0, :2]
y = y[y != 0]

# split train test
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

# %%
# Q1 Linear kernel

# Définir la grille de paramètres pour le noyau linéaire
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

# Utiliser GridSearchCV pour trouver les meilleurs paramètres avec un noyau linéaire
clf_linear = GridSearchCV(SVC(), parameters, cv=5)
clf_linear.fit(X_train, y_train)

# Calculer la précision du modèle
y_pred_train = clf_linear.predict(X_train)
y_pred_test = clf_linear.predict(X_test)

print('Meilleur paramètre C pour le noyau linéaire :', clf_linear.best_params_)
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

# %%
# Q2 polynomial kernel

# Définir la grille de paramètres pour le noyau polynomial
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[ 1,2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

# Utiliser GridSearchCV pour le noyau polynomial
clf_poly = GridSearchCV(SVC(), parameters, cv=5)
clf_poly.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print(clf_poly.best_params_)

# Calculer la précision du modèle polynomial
y_pred_train_poly = clf_poly.predict(X_train)
y_pred_test_poly = clf_poly.predict(X_test)

print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


# %%
# display your results using frontiere

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

def plot_2d(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def frontiere(f, X, y, step=50):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step), np.linspace(y_min, y_max, step))
    Z = np.array([f(np.array([xx_, yy_])) for xx_, yy_ in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='autumn')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)

plt.ion()
plt.figure(figsize=(15, 5))

plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)
plt.title("polynomial kernel")

plt.tight_layout()
plt.show()

# %%
