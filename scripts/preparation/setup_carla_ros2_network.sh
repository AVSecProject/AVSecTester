#!/usr/bin/env bash
# Create the user's local DDS bridge and print the Compose settings it needs.
set -euo pipefail

if [[ $# != 1 || ! "$1" =~ ^(0|[1-9][0-9]{0,2})$ ]] || (( $1 > 232 )); then
  echo "Usage: $0 ROS_DOMAIN_ID (an integer from 0 to 232)" >&2
  exit 2
fi
domain=$1
script_directory=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(cd -- "$script_directory/../.." && pwd -P)
source "$script_directory/_pinned_carla_image.sh"
image=$(pinned_carla_image "$repository")
if ! image_id=$(docker image inspect --format '{{.Id}}' "$image"); then
  echo "The pinned CARLA image must already be available locally: $image" >&2
  exit 1
fi

uid=$(id -u)
network="avsectester-ros2-$uid"
keepalive="avsectester-ros2-keepalive-$uid"
owner_label="avsectester.ros2-user=$uid"
created_network=
created_container=
temporary=
cleanup() {
  local status=$?
  if [[ -n "$temporary" ]]; then
    rm -rf -- "$temporary"
  fi
  if (( status != 0 )); then
    if [[ -n "$created_container" ]]; then
      docker rm -f "$created_container" >&2 || true
    fi
    if [[ -n "$created_network" ]]; then
      docker network rm "$created_network" >&2 || true
    fi
  fi
}
trap cleanup EXIT

if network_info=$(docker network inspect --format \
    '{{.Id}} {{.Driver}} {{index .Labels "avsectester.ros2-user"}}' "$network" 2>/dev/null); then
  read -r network_id driver owner <<< "$network_info"
  if [[ "$driver" != bridge || "$owner" != "$uid" ]]; then
    echo "Network $network is not this user's AVSecTester ROS 2 bridge." >&2
    exit 1
  fi
else
  network_id=$(docker network create --driver bridge --label "$owner_label" "$network")
  created_network=$network_id
fi

if container_info=$(docker container inspect --format \
    '{{.Id}} {{index .Config.Labels "avsectester.ros2-user"}} {{.Image}} {{.HostConfig.RestartPolicy.Name}} {{.State.Running}}' \
    "$keepalive" 2>/dev/null); then
  read -r container_id owner current_image restart_policy running <<< "$container_info"
  network_format="{{with index .NetworkSettings.Networks \"$network\"}}{{.NetworkID}}{{end}}"
  container_network=$(docker container inspect --format "$network_format" "$container_id")
  if [[ "$owner" != "$uid" || "$current_image" != "$image_id" ||
        "$restart_policy" != unless-stopped || "$container_network" != "$network_id" ]]; then
    echo "Container $keepalive does not match this user's AVSecTester ROS 2 bridge settings." >&2
    exit 1
  fi
else
  container_id=$(docker create --pull=never --label "$owner_label" --name "$keepalive" \
    --network "$network_id" --restart unless-stopped -e NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint sleep "$image" infinity)
  created_container=$container_id
  running=false
fi
if [[ "$running" != true ]]; then
  docker start "$container_id" >&2
fi

bridge="br-${network_id:0:12}"
link=
for _ in {1..15}; do
  link=$(ip -o link show dev "$bridge" 2>/dev/null || true)
  if [[ "$link" == *" state UP "* ]]; then
    break
  fi
  sleep .2
done
if [[ "$link" != *" state UP "* ]]; then
  echo "Docker network $network has no UP host bridge at $bridge." >&2
  exit 1
fi
bridge_ip=$(ip -o -4 address show dev "$bridge" scope global |
  awk 'NR == 1 { split($4, address, "/"); print address[1] }')
if [[ -z "$bridge_ip" ]]; then
  echo "No IPv4 address on Docker bridge $bridge." >&2
  exit 1
fi

dds="$repository/data/carla_ros2"
mkdir -p -- "$dds"
temporary=$(mktemp -d "$dds/.setup.XXXXXX")

# Use the official CARLA Autoware launcher's UDP transports and receive buffer sizing.
cat > "$temporary/fastdds_profile.xml" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>udp_bridge_only</transport_id>
            <type>UDPv4</type>
            <interfaceWhiteList>
                <address>$bridge_ip</address>
            </interfaceWhiteList>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="carla_bridge_only" is_default_profile="true">
        <rtps>
            <useBuiltinTransports>false</useBuiltinTransports>
            <userTransports>
                <transport_id>udp_bridge_only</transport_id>
            </userTransports>
        </rtps>
    </participant>
</profiles>
XML
cat > "$temporary/cyclonedds.xml" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain Id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="$bridge" priority="default" multicast="default"/>
            </Interfaces>
            <AllowMulticast>default</AllowMulticast>
            <MaxMessageSize>65500B</MaxMessageSize>
        </General>
        <Discovery>
            <MaxAutoParticipantIndex>300</MaxAutoParticipantIndex>
        </Discovery>
        <Internal>
            <SocketReceiveBufferSize min="10MB" max="64MB"/>
            <Watermarks>
                <WhcHigh>500kB</WhcHigh>
            </Watermarks>
        </Internal>
    </Domain>
</CycloneDDS>
XML
mv -f -- "$temporary/fastdds_profile.xml" "$temporary/cyclonedds.xml" "$dds/"
echo "DDS bridge: $bridge ($bridge_ip), owner: $uid" >&2
printf "CARLA_SERVER_ARGS='-ros2 -rmw=fastdds -ros-domain-id=%s'\n" "$domain"
printf "FASTRTPS_DEFAULT_PROFILES_FILE='/dds/fastdds_profile.xml'\n"
printf "CARLA_FASTDDS_PROFILE_HOST='%s'\n" "$(printf '%s' "$dds/fastdds_profile.xml" | sed "s/'/\\\\\\'/g")"
