# TP3 - Support Vector Machine (SVM)

## Description du projet

Ce dépôt contient le rapport du TP3 sur les **Support Vector Machines (SVM)**, réalisé dans le cadre du **M2 MIND-SIAD** à l'Université de Montpellier. Ce rapport couvre plusieurs aspects des SVM, y compris l'implémentation et l'évaluation de modèles SVM à travers des scénarios variés.

Le TP explore :

- Les fondements théoriques des SVM
- Classification avec les SVM sur le jeu de données Iris
- Impact du noyau linéaire et du paramètre \(C\) sur des jeux de données déséquilibrés
- Classification de visages
- Réduction de dimension via la PCA

## Structure du projet

Voici la structure du projet :


- **image/** : Ce dossier contient les images utilisées dans le fichier LaTeX `tp.tex`. Elles doivent être importées sur Overleaf pour garantir une compilation correcte.
- **src/** : Ce dossier contient les fichiers Python (`.py`) utilisés pour entraîner et tester les modèles SVM. Vous pouvez exécuter ces scripts pour reproduire les résultats du rapport.\
  svm_iris.py: Ce script contient le code de classification pour les données Iris.\
  svm_face.py : Ce script contient le code pour la classification de visages.

- **venv_app/** : Contient l'environnement virtuel Python avec les dépendances requises pour les scripts.
- **tp.tex** : Fichier LaTeX contenant le rapport complet.
- **tp_apprentissage_stat.pdf** : Version PDF générée du rapport LaTeX.
- **.gitignore** : Fichier définissant les fichiers et dossiers à ignorer dans le dépôt Git
## Générer le rapport sur Overleaf

Pour compiler le fichier LaTeX `tp.tex` et générer le rapport en PDF sur Overleaf, voici les étapes à suivre :

1. **Téléchargez les fichiers** : Téléchargez tout le dépôt, incluant :
   - `tp.tex` : Le fichier LaTeX principal
   - Le dossier **image/** qui contient les images utilisées dans `tp.tex`
   - Le dossier **src/** contenant les scripts Python (si vous souhaitez les exécuter pour reproduire les résultats)

2. **Accédez à Overleaf** : Connectez-vous sur [Overleaf](https://www.overleaf.com) et cliquez sur "Nouveau projet" > "Import project".

3. **Importer les fichiers sur Overleaf** :
   - Téléchargez **tout le contenu du projet** : `tp.tex`, le dossier `image/` et tous les autres fichiers nécessaires à la compilation.
   - Assurez-vous que le dossier `image/` soit bien importé, car Overleaf doit y accéder pour insérer les images dans le document PDF.

4. **Compiler le document** :
   - Dans Overleaf, sélectionnez `tp.tex` comme fichier principal.
   - Cliquez sur "Recompiler" pour générer le fichier PDF.

5. **Télécharger le PDF** :
   - Une fois la compilation réussie, vous pourrez télécharger le fichier PDF généré depuis Overleaf.

## Instructions pour l'exécution des scripts

Les scripts Python utilisés dans ce TP sont disponibles dans le dossier `src/`. Pour exécuter ces scripts localement :

1. **Installer l'environnement virtuel** (optionnel) :
   - Si vous utilisez un environnement virtuel, vous pouvez l'activer avec `source venv_app/bin/activate` sur Linux/MacOS ou `venv_app\Scripts\activate` sur Windows.

2. **Installer les dépendances** :
   - Installez les bibliothèques nécessaires avec la commande :
     ```bash
     pip install -r requirements.txt
     ```

3. **Exécuter les scripts** :
   - Les scripts Python se trouvent dans le dossier `src/`. Vous pouvez les exécuter avec :
     ```bash
     python src/votre_script.py
     ```

## Auteurs

- **HAMOMI Majda** - M2 MIND-SIAD



