# Shared by the CARLA preparation commands.
pinned_carla_image() {
  local image
  image=$(awk '$0 == "  carla-server:" { carla = 1; next }
               carla && /^  [^ ]/ { exit }
               carla && /^    image: / { print $2; exit }' "$1/docker-compose.yml")
  if [[ ! "$image" =~ ^carlasim/carla:[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
    echo "No pinned CARLA server image in $1/docker-compose.yml" >&2
    return 1
  fi
  printf '%s\n' "$image"
}
