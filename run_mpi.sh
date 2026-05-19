#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Script de benchmark para multiplicación de matrices con MPI
#
# Ejecuta la versión MPI sobre el clúster:
#   head, wn1, wn2, wn3
#
# Programa esperado:
#   ./matmul_mpi N trials [seed]
#
# Ejecución:
#   mpirun -np P --host ... ./matmul_mpi N 1 seed
#
# Salida esperada del programa:
#   N processes trial wall_s checksum seed
# ============================================================

DIR="$(cd "$(dirname "$0")" && pwd)"

SRC="$DIR/matmul_mpi.c"
BIN="$DIR/matmul_mpi"

OUT_DIR="$DIR/output"
OUT="$OUT_DIR/output_mpi.txt"

SEED=123456789

# Número de repeticiones externas.
ROUNDS=4

# Tamaños de matriz.
# Importante: como vamos a probar hasta 8 procesos, N debe ser divisible entre 8.
SIZES=(400 600 800 1000 1200 1600)


# Procesos MPI a evaluar.
# 1 proceso  -> solo head
# 2 procesos -> head + wn1
# 4 procesos -> head + wn1 + wn2 + wn3
# 8 procesos -> 2 procesos por nodo
PROCS=(1 2 4 8)

# Nodos del clúster.
WORKERS=(wn1 wn2 wn3)

mkdir -p "$OUT_DIR"

if [ ! -f "$SRC" ]; then
  echo "Error: no se encontró el código fuente: $SRC" >&2
  exit 1
fi

echo "Compilando versión MPI..."
mpicc "$SRC" -o "$BIN"
echo "Compilación MPI finalizada."

echo ""
echo "Copiando binario MPI a nodos trabajadores..."

for node in "${WORKERS[@]}"; do
  echo "  Copiando a $node..."
  scp "$BIN" "hpcuser@$node:$BIN"
done

echo "Copia finalizada."

# Encabezado
echo "N processes trial wall_s checksum seed" > "$OUT"

echo ""
echo "Ejecutando benchmark MPI..."
echo "Salida: $OUT"
echo ""

for round in $(seq 1 "$ROUNDS"); do
  echo "Ronda MPI $round/$ROUNDS..."

  for N in "${SIZES[@]}"; do
    for P in "${PROCS[@]}"; do

      echo "  N=$N procesos=$P"

      if [ "$P" -eq 1 ]; then
        HOSTS="head:1"
      elif [ "$P" -eq 2 ]; then
        HOSTS="head:1,wn1:1"
      elif [ "$P" -eq 4 ]; then
        HOSTS="head:1,wn1:1,wn2:1,wn3:1"
      elif [ "$P" -eq 8 ]; then
        # 8 procesos en 4 nodos: 2 procesos por nodo.
        HOSTS="head:2,wn1:2,wn2:2,wn3:2"
      else
        echo "Número de procesos no soportado en este script: $P" >&2
        exit 1
      fi

      mpirun \
        -np "$P" \
        --host "$HOSTS" \
        "$BIN" "$N" 1 "$SEED" | awk 'NR > 1' >> "$OUT"

    done
  done
done

echo ""
echo "Benchmark MPI completado."
echo "Resultados guardados en: $OUT"