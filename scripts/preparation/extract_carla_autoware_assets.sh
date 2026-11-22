#!/usr/bin/env bash
# Extract and verify the official CARLA Autoware assets and camera profile from the pinned image.
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi
script_directory=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(cd -- "$script_directory/../.." && pwd -P)
source "$script_directory/_pinned_carla_image.sh"
image=$(pinned_carla_image "$repository")
if ! docker image inspect "$image" >/dev/null; then
  echo "The pinned CARLA image must already be available locally: $image" >&2
  exit 1
fi

assets="$repository/data/carla_autoware"
mkdir -p -- "$repository/data"
staging=$(mktemp -d "$repository/data/.carla_autoware.XXXXXX")
container=
cleanup() {
  local status=$?
  if [[ -n "$container" ]]; then
    docker rm "$container" >&2 || status=1
  fi
  rm -rf -- "$staging"
  exit "$status"
}
trap cleanup EXIT
container=$(docker create --pull=never --label "avsectester.asset-user=$(id -u)" \
  --entrypoint sleep "$image" infinity)
subdirectory=/workspace/PythonAPI/examples/av_stacks/autoware
while read -r checksum relative; do
  file="$staging/$relative"
  mkdir -p -- "$(dirname -- "$file")"
  docker cp "$container:$subdirectory/$relative" "$file"
  printf '%s  %s\n' "$checksum" "$file" | sha256sum --check --strict
done <<'ASSETS'
326ef4585bf459be07f593d41acff237622e769cf1a373d9bed4f69653d37dc0 run/carla_traffic_light_state_source.py
8637b870e69e73ddc6cdbf45a980a278b7d3f3f04ebfcb5119d82673acf05258 run/autoware_traffic_light_state_publisher.py
ed06dba0dc480504300e1afbefbddec44fc9e38fef75416fc5d0957971ccd052 map_tools/maps/Town10HD/lanelet2_map.osm
ed8c0db03ceb21ee058b2bee225ca61501b7ddf5a185c4efe6be6911b8e4673e map_tools/maps/Town10HD/lanelet2_map_2019_legacy.osm
84c7f9f0acd71778fe6cd541bebb28fcba108a4e3ba0937edae323ba451f8697 map_tools/maps/Town10HD/map_projector_info.yaml
e00f40ae2a6b3f1e3277cba6c8a1d4428452daff84aab2929b08220b118b2f01 map_tools/maps/Town10HD/pointcloud_map.pcd
ASSETS

# The official demo writes its camera profile from these settings before it connects. A
# packaged server reads the profile from its content directory, which
# docker-compose.autoware.yml mounts.
profile="$staging/postprocess/autoware_demo.json"
mkdir -p -- "$(dirname -- "$profile")"
docker run --rm --pull=never --network none -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python3 "$image" -I -c "$(cat <<'PYTHON'
import ast
import json
import sys

PATH = "/workspace/PythonAPI/examples/autoware_demo.py"
tree = ast.parse(open(PATH, encoding="utf-8").read(), PATH)
for node in tree.body:
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "AUTOWARE_POSTPROCESS_SETTINGS"):
        json.dump(ast.literal_eval(node.value), sys.stdout, indent=2)
        break
else:
    sys.exit(f"AUTOWARE_POSTPROCESS_SETTINGS not found in {PATH}")
PYTHON
)" > "$profile"
printf '%s  %s\n' 0b1fd3fa6cbee98a4ce28a38ab22da81b16882b2d4d71e475e833b33d0868a31 "$profile" |
  sha256sum --check --strict

mkdir -p -- "$assets"
cp -a -- "$staging/." "$assets/"
echo "Verified official CARLA Autoware assets and camera profile in $assets"
