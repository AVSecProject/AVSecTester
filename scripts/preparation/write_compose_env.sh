#!/usr/bin/env bash
# Write machine-specific Compose settings beside the repository's compose file.
set -euo pipefail
force=false
if [[ $# -eq 1 && "$1" == --force ]]; then
  force=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
destination="$repo_root/.env"
if [[ -e "$destination" && "$force" != true ]]; then
  echo "Refusing to overwrite $destination; use --force." >&2
  exit 1
fi
docker_cli=$(command -v docker) || { echo "docker is not on PATH." >&2; exit 1; }
docker_gid=$(stat -c %g /var/run/docker.sock)
compose_uid=$(id -u)
compose_gid=$(id -g)
temporary=$(mktemp "$repo_root/.env.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
# Single-quoted dotenv values preserve spaces and literal dollar signs in paths.
dotenv_path() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/\\\\'/g")"
}
{
  printf 'AVSECTESTER_ROOT=%s\n' "$(dotenv_path "$repo_root")"
  printf 'AVSECTESTER_UID=%s\n' "$compose_uid"
  printf 'AVSECTESTER_GID=%s\n' "$compose_gid"
  printf 'DOCKER_GID=%s\n' "$docker_gid"
  printf 'DOCKER_CLI=%s\n' "$(dotenv_path "$docker_cli")"
  printf 'CARLA_SERVER_GPU=0\nCARLA_EXPERIMENT_GPU=0\nNUREC_GPU=0\n'
} > "$temporary"
mv -- "$temporary" "$destination"
cat -- "$destination"
