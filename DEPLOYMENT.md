# Déploiement — luckyIA

Ce document explique comment construire, lancer et déployer l'application luckyIA (Flask + ai_module) en utilisant Docker, docker-compose et des bonnes pratiques de production.

Langue : Français

---

## Récapitulatif rapide

- Image Docker fournie via `Dockerfile` à la racine.
- Compose simplifié fourni via `docker-compose.yml` (service `web` exposant le port 7000).
- Script d'aide `deploy/docker-build-and-run.sh` pour construire et démarrer un conteneur localement.
- Par défaut l'image est construite sans PyTorch (`INSTALL_TORCH=false`) pour éviter les problèmes de compatibilité sur certaines plateformes. Passez `--with-torch` au script si vous voulez tenter l'installation de `torch` dans l'image.

---

## Prérequis

- Docker (Engine) installé
- (Optionnel) docker-compose
- Accès à la racine du dépôt

Sur macOS / Linux, installez Docker Desktop ou docker engine via votre gestionnaire.

---

## Structure des fichiers utiles

- `Dockerfile` — le Dockerfile principal.
- `docker-compose.yml` — compose pour développement local.
- `deploy/docker-build-and-run.sh` — script pratique pour build + run.
- `.dockerignore` — ignore les fichiers inutiles pendant le build.
- `web_app/README_docker.md` — mini-guide spécifique à la web app.
- `DEPLOYMENT.md` — ce document.

---

## Construire l'image Docker

Depuis la racine du projet vous pouvez construire l'image de deux façons.

1) Sans PyTorch (recommandé pour la plupart des environnements de développement) :

```bash
# depuis la racine du projet
docker build --build-arg INSTALL_TORCH=false -t luckyia:latest .
```

2) Avec PyTorch (optionnel — image plus lourde et peut échouer si roue non disponible pour la plateforme) :

```bash
# tentative d'installation de torch CPU dans l'image
docker build --build-arg INSTALL_TORCH=true -t luckyia:latest .
```

Notes :
- L'option `INSTALL_TORCH=true` tente d'installer une version CPU de `torch` avant d'installer le reste des dépendances.
- Si le build échoue à cause de PyTorch, reconstruisez sans `torch` puis installez `torch` séparément sur une machine compatible.

---

## Lancer le conteneur

Exécuter le conteneur (mount du code pour dev) :

```bash
docker run -d --name luckyia_web -p 7000:7000 -v "$(pwd)":/app:rw luckyia:latest
```

- L'application Flask écoute sur le port 7000 par défaut (`web_app/app.py`).
- Le volume `-v "$(pwd)":/app` permet d'éditer le code hors conteneur et voir les changements (pratique pour dev). En production, préférez ne PAS monter le code en volume.

Pour suivre les logs :

```bash
docker logs -f luckyia_web
```

Pour arrêter et supprimer :

```bash
docker rm -f luckyia_web
```

---

## Utiliser `docker-compose`

Fichier `docker-compose.yml` fourni permet de démarrer le service facilement :

```bash
docker-compose up --build
```

Cela expose aussi le port 7000 et monte le volume courant.

---

## Script d'aide (déployé)

Un script d'aide est fourni : `deploy/docker-build-and-run.sh`.

Usage :

```bash
# sans torch (par défaut)
./deploy/docker-build-and-run.sh

# tenter d'installer torch pendant le build (peut échouer selon la plateforme)
./deploy/docker-build-and-run.sh --with-torch
```

Le script :
- construit l'image,
- arrête et supprime un conteneur existant nommé `luckyia_web`,
- démarre un nouveau conteneur en arrière-plan.

---

## Configuration via variables d'environnement

L'image supporte plusieurs variables d'environnement (définies dans le Dockerfile ou via `docker run -e` / compose) :

- `FLASK_APP` (par défaut `web_app/app.py`)
- `FLASK_RUN_HOST` (par défaut `0.0.0.0`)
- `FLASK_RUN_PORT` (par défaut `7000`)

Exemple :

```bash
docker run -d -p 7000:7000 -e FLASK_APP=web_app/app.py -e FLASK_RUN_PORT=7000 luckyia:latest
```

Pour la production, vous pouvez définir d'autres variables (clé secrète, mode, configuration DB) via un fichier `.env` et passer `--env-file` à `docker run` ou `env_file` dans compose.

---

## Recommandations pour la production

1) Ne PAS exécuter Flask en mode serveur intégré (Werkzeug) en production. Utilisez gunicorn ou uwsgi.

  - Exemple Dockerfile de production (résumé) :

  - Utilisez une image multi-stage : builder installe deps, runtime copie uniquement artefacts.
  - Installer `gunicorn` dans l'image et exécuter : `gunicorn -b 0.0.0.0:7000 "web_app.app:app" --workers 3`

2) Ne pas monter le code en volume en production — copiez le code dans l'image et démarrez le conteneur immuable.

3) Utiliser un reverse-proxy (Nginx) devant gunicorn pour gérer TLS, compression et buffering.

4) Stockage des modèles et données :
  - Les modèles sont par défaut dans le dossier `models/` du dépôt. En production, stockez-les dans un volume persistant ou un storage externe (S3, NFS, PVC). Montez un volume Docker ou utilisez un service de stockage.

5) Secrets et configuration :
  - NE PAS stocker de secrets dans le code ni dans les sidecars meta en clair. Utilisez variables d'environnement, secrets manager, ou `docker secret`.

---

## Optimisations avancées

- Image multi-stage pour réduire taille finale.
- Pinning plus strict des versions Python/paquets dans `requirements.txt`.
- Construire artefacts wheel pour dépendances lourdes en amont et les héberger dans un index interne.
- Si `torch` est nécessaire, préférez construire sur une base compatible (ex : image officielle PyTorch, ou builder stage sur `nvidia/cuda` si GPU requis).

---

## CI / CD (exemple GitHub Actions)

Voici un exemple minimal pour builder et pousser l'image vers un registry Docker Hub / GitHub Container Registry.

```yaml
name: Build and push Docker image

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/luckyia:latest
          build-args: |
            INSTALL_TORCH=false
```

Notes : adaptez `push` et `registry` selon votre cible.

---

## Debug & dépannage

- Si build échoue pendant `pip install torch` :
  - Vérifiez la version de Python utilisée par le Dockerfile (ici `python:3.11-slim`). PyTorch n'a parfois pas de roue pour toutes versions.
  - Essayez sans torch (option par défaut), puis installez torch séparément sur une machine compatible.

- Logs d'exécution :
  - `docker logs -f luckyia_web` pour suivre la sortie.
  - `docker exec -it luckyia_web /bin/bash` pour un shell interactif (si image contient bash).

- Vérifier que le port est exposé et non occupé par un autre service.

---

## Commandes utiles de maintenance

Nettoyage images & conteneurs :

```bash
# lister conteneurs
docker ps -a
# supprimer conteneurs
docker rm -f luckyia_web || true
# supprimer image
docker rmi luckyia:latest || true
# supprimer images dangling
docker image prune -f
```

---

## Checklist rapide avant mise en production

- [ ] Construire image en mode production (sans montages), intégrer `gunicorn`.
- [ ] Déployer modèles sur volume persistant ou storage externe.
- [ ] Mettre en place TLS et reverse proxy (Nginx).
- [ ] Mettre en place monitoring & logs (prometheus, grafana, ELK/EFK).
- [ ] Tests d'intégration end-to-end.

---

Si vous souhaitez, je peux :

- générer un Dockerfile multi-stage optimisé pour la production, ou
- ajouter un service `nginx` et un `docker-compose.prod.yml` d'exemple, ou
- ajouter le snippet `systemd`/unit pour déployer le conteneur sur un serveur.

Dites-moi quelle option vous préférez et je l'implémente. 

