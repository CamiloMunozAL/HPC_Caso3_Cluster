#!/usr/bin/env bash
set -euo pipefail

DIR="$HOME/HPC_Caso3_Cluster"
WORKERS=(wn1 wn2 wn3)

if [ ! -d "$DIR" ]; then
  echo "No existe el directorio: $DIR" >&2
  exit 1
fi

for node in "${WORKERS[@]}"; do
  echo "Sincronizando con $node..."
  ssh "hpcuser@$node" "mkdir -p '$DIR'"
  rsync -av --delete "$DIR/" "hpcuser@$node:$DIR/"
done

echo "Sincronización completada."