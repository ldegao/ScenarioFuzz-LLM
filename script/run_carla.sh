#!/bin/bash

# Find the most idle GPU
idle_gpu=0
port=4000
carla_cmd="./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=$port -quality-level=Epic && /bin/bash"

# Get project root directory (assuming script is in script/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RECORDER_DIR="$PROJECT_ROOT/data/output/recorder"

# Create recorder directory if it doesn't exist
mkdir -p "$RECORDER_DIR"
# Set permissions to allow CARLA container to write recorder files
# CARLA runs as user 998 (carla), so we need to ensure the directory is writable
chmod 777 "$RECORDER_DIR"

docker run --name="carla-$USER" \
  -d \
  --gpus "device=$idle_gpu" \
  --net=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$RECORDER_DIR:/home/carla/recordings:rw" \
  carlasim/carla:0.9.13 \
  $carla_cmd
