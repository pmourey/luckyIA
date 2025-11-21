#!/usr/bin/env bash
# Build and run the luckyIA docker image and container.
# Usage: ./deploy/docker-build-and-run.sh [--with-torch]

set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Quick check: is Docker daemon reachable? Give actionable error if not.
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' CLI not found. Please install Docker Desktop or Docker Engine and ensure 'docker' is on PATH." >&2
  exit 1
fi

# Print DOCKER_HOST if set (common cause of remote/incorrect daemon address)
if [ -n "${DOCKER_HOST:-}" ]; then
  echo "Notice: DOCKER_HOST is set to '$DOCKER_HOST' — this may point to a remote daemon." >&2
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Cannot connect to the Docker daemon." >&2
  echo "Possible causes:" >&2
  echo "  * Docker Desktop / Docker Engine is not running." >&2
  echo "  * DOCKER_HOST is set to a wrong/remote address (e.g. tcp://192.168.1.200:2375)." >&2
  echo "  * Permission issue: your user cannot access the Docker socket (on Linux)." >&2
  echo "Quick fixes:" >&2
  echo "  - Start Docker Desktop (macOS/Windows) or start the Docker service: sudo systemctl start docker (Linux)." >&2
  echo "  - If DOCKER_HOST is set but you want the local daemon: run 'unset DOCKER_HOST' or remove it from your shell profile." >&2
  echo "  - Check 'docker context ls' and use 'docker context use default' if needed." >&2
  echo "  - Run 'docker info' to see the exact error message." >&2
  echo "Example commands:" >&2
  echo "  docker info" >&2
  echo "  echo \$DOCKER_HOST" >&2
  echo "  unset DOCKER_HOST" >&2
  exit 1
fi

BUILD_ARG="--build-arg INSTALL_TORCH=false"
if [ "${1:-}" = "--with-torch" ]; then
  BUILD_ARG="--build-arg INSTALL_TORCH=true"
fi

IMAGE_NAME=luckyia:latest
CONTAINER_NAME=luckyia_web

# Build
echo "Building Docker image ($IMAGE_NAME) with arg: $BUILD_ARG"
docker build $BUILD_ARG -t $IMAGE_NAME .

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "Stopping and removing existing container ${CONTAINER_NAME}"
  docker rm -f ${CONTAINER_NAME} || true
fi

# Run
echo "Running container ${CONTAINER_NAME}"
docker run -d --name ${CONTAINER_NAME} -p 7000:7000 -v "$ROOT_DIR":/app:rw ${IMAGE_NAME}

echo "Container started. App should be available at http://127.0.0.1:7000"

echo "To follow logs: docker logs -f ${CONTAINER_NAME}"
