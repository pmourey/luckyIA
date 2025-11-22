# luckyIA

Module d'Intelligence Artificielle en Python pour charger, entraîner et servir des modèles à partir de données clients.

Ce dépôt contient :
- un module IA réutilisable `ai_module/` (chargement, prétraitement, modèle PyTorch, entraînement),
- une application web légère Flask dans `web_app/` pour téléverser des CSV, choisir un modèle et obtenir des prédictions,
- des exemples et utilitaires dans `examples/` pour générer des données et entraîner des modèles.

---

## Points clés
- Target principal : prédire une colonne `risque` (variable cible) à partir de features client.
- Interface web : fournie dans `web_app/` (page d'accueil, génération de données/modèles, entraînement simple).
- Déploiement : Docker et docker-compose fournis (voir `DEPLOYMENT.md` pour les détails).

---

## Installation locale (développement)

1. Créez et activez un environnement Python :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

Remarque : l'installation de `torch` peut échouer selon votre plateforme/Python. Si nécessaire, installez `torch` séparément en suivant les instructions officielles : https://pytorch.org

---

## Lancer localement (Flask)

L'application web démarre par défaut sur le port **7000** (configuration dans `web_app/app.py`).

```bash
# depuis la racine du projet
cd web_app
python app.py
```

Ouvrez ensuite : http://127.0.0.1:7000/

Notes :
- L'UI permet de charger un CSV (sans la colonne `risque` pour l'inférence), choisir un modèle dans `models/` et afficher les résultats.
- Si le CSV contient la colonne `risque`, l'application essaie de la détecter et de la retirer automatiquement (cette action est loggée et un message est affiché).

---

## Docker (guide rapide)

Un `Dockerfile` et deux fichiers Compose sont fournis :
- `docker-compose.yml` (développement, monte le code en volume et expose le port 7000),
- `docker-compose.prod.yml` (production : Gunicorn + Nginx, expose le port 80).

Construction et exécution rapide (sans PyTorch) :

```bash
# depuis la racine
docker build --build-arg INSTALL_TORCH=false -t luckyia:latest .
docker run -d --name luckyia_web -p 7000:7000 -v "$(pwd)":/app:rw luckyia:latest
```

Pour la production (Gunicorn + Nginx) :

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Consultez `DEPLOYMENT.md` pour la documentation complète et les conseils d'exploitation.

---

## Tests

Exécutez la suite de tests unitaires avec pytest :

```bash
python -m pytest -q
```

---

## La colonne `risque` (target)

- `risque` est la variable cible utilisée pour l'entraînement. Formats acceptés :
  - binaire (0/1),
  - probabilité continue (0.0–1.0),
  - score numérique (nécessite normalisation si différent de l'intervalle attendu).
- Pour l'entraînement, fournissez un CSV contenant `risque`.
- Pour l'inférence, fournissez un CSV avec les seules features (sans `risque`). L'UI détecte et propose de retirer `risque` si présent.

---

## Variables d'environnement utiles

- `FLASK_SECRET` — clé secrète de Flask (par défaut `dev-secret`).
- `GITHUB_REPO_URL` — si défini, un lien vers le dépôt sera affiché dans le footer de l'UI.
- `INSTALL_TORCH` — argument de build Docker (voir Dockerfile) pour tenter d'installer `torch` pendant le build.

---

## Contribution

Les contributions sont les bienvenues : issues, forks et PRs sont appréciés. Consultez le code et les tests pour comprendre les comportements attendus.

---

## Ressources

- Documentation déploiement : `DEPLOYMENT.md`
- Web app : `web_app/`
- Module IA : `ai_module/`

---

Si vous voulez que j'ajoute un badge, un lien direct vers un fichier de licence ou un exemple de workflow CI (GitHub Actions) dans le README, dites-moi lequel et je l'ajoute.
