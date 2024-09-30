
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')


###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)


# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()


#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set
























# %%
# Q3 - Compléter cette partie
from sklearn.metrics import accuracy_score

print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []

for C in Cs:
    # Créer un modèle SVM avec un noyau linéaire et le paramètre de régularisation C
    clf = SVC(kernel='linear', C=C)
    
    # Entraîner le modèle sur l'ensemble d'entraînement
    clf.fit(X_train, y_train)
    
    # Prédire les labels sur l'ensemble de test
    y_pred = clf.predict(X_test)
    
    # Calculer le score (accuracy) et l'ajouter à la liste des scores
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

# Trouver la valeur de C qui donne le meilleur score
ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

# Afficher la courbe des scores en fonction de C
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Paramètres de régularisation C")
plt.ylabel("Scores d'apprentissage (accuracy)")
plt.xscale("log")
plt.tight_layout()
plt.show()

print("Best score: {}".format(np.max(scores)))

# Prédire les noms des personnes dans l'ensemble de test avec le meilleur modèle
print("Predicting the people names on the testing set")
t0 = time()

# Utiliser la meilleure valeur de C pour prédire
best_clf = SVC(kernel='linear', C=Cs[ind])
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

# Afficher le score final avec le meilleur C
final_score = accuracy_score(y_test, y_pred_best)
print("Final accuracy with best C: {:.2f}".format(final_score))


#%% 
# predict labels for the X_test images with the best classifier
clf = best_clf  # On utilise le meilleur classificateur trouvé précédemment

t0 = time()
y_pred = clf.predict(X_test)  # Prédiction des labels
print("done in %0.3fs" % (time() - t0))

# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))  # Accuracy du meilleur modèle



















#%% 
####################################################################
# Qualitative evaluation of the predictions using matplotlib

# Fonction pour générer des titres selon la prédiction et la vérité terrain
def title(pred, true, names):
    pred_name = names[int(pred)]
    true_name = names[int(true)]
    return f"Pred: {pred_name}\nTrue: {true_name}"

# Créer les titres basés sur les prédictions
prediction_titles = [title(y_pred[i], y_test[i], names) for i in range(y_pred.shape[0])]

# Afficher la galerie d'images avec les prédictions
plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients (for a linear kernel)

if clf.kernel == 'linear':  # Ce plot fonctionne uniquement pour un noyau linéaire
    plt.figure()
    # Afficher les coefficients sous forme d'image (correspond à l'importance des pixels)
    plt.imshow(np.reshape(clf.coef_[0], (h, w)), interpolation='nearest', cmap=plt.cm.hot)
    plt.title("Coefficient image (importance des pixels)")
    plt.colorbar()
    plt.show()
else:
    print("Le modèle n'est pas linéaire, donc les coefficients ne peuvent pas être affichés.")

# %% 

#%% 
# Q4Q4 Ajout des variables de nuisances et comparaison de la performance

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    # Paramètres pour SVM linéaire avec C variant sur une plage logarithmique
    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    # Calcul des scores sur les données d'entraînement et de test
    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

# Exécution du SVM sur les données sans nuisance
print("Score sans variable de nuisance")
run_svm_cv(X, y)

# Ajout des variables de nuisance (bruit)
n_features = X.shape[1]
sigma = 1
# Génération de 300 variables de nuisance avec une distribution gaussienne
noise = sigma * np.random.randn(n_samples, 300)
X_noisy = np.concatenate((X, noise), axis=1)  # Ajout des variables de nuisance aux données originales
X_noisy = X_noisy[np.random.permutation(X.shape[0])]  # Permutation aléatoire des données pour ne pas biaiser les résultats

# Exécution du SVM sur les données avec variables de nuisance
print("Score avec variables de nuisance")
run_svm_cv(X_noisy, y)



#%%
# Q5
print("Score après réduction de dimension avec PCA")

n_components = 100 # Nombre de composantes principales (peut être ajusté)
pca = PCA(n_components=n_components, svd_solver='randomized').fit(X_noisy)

# Transformation des données avec PCA
X_noisy_pca = pca.transform(X_noisy)

# On affiche la variance expliquée pour choisir le bon nombre de composantes
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"Variance expliquée avec {n_components} composantes : {explained_variance:.2%}")








# %%
