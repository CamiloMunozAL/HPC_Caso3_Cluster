#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Script de benchmark para multiplicación de matrices secuencial
#
# Ejecuta la versión secuencial únicamente en el nodo head.
#
# Programa esperado:
#   ./matmul_seq N trials [seed]
#
# Salida esperada del programa:
#   N version trial wall_s user_s kernel_s cpu_total_s checksum seed
# ============================================================

DIR="$(cd "$(dirname "$0")" && pwd)"

SRC="$DIR/matmul_seq.c"
BIN="$DIR/matmul_seq"

OUT_DIR="$DIR/output"
OUT="$OUT_DIR/output_seq.txt"

SEED=123456789

# Número de repeticiones externas del script.
# Cada ejecución del programa usará trials=1 para que cada fila sea una medición.
ROUNDS=4

# Tamaños recomendados inicialmente.
# Todos son seguros para pruebas iniciales en e2-small.
SIZES=(400 600 800 1000 1200 1600)


mkdir -p "$OUT_DIR"

if [ ! -f "$SRC" ]; then
  echo "Error: no se encontró el código fuente: $SRC" >&2
  exit 1
fi

echo "Compilando versión secuencial..."
gcc -Wall -Wextra "$SRC" -o "$BIN"
echo "Compilación secuencial finalizada."

# Encabezado
echo "N version trial wall_s user_s kernel_s cpu_total_s checksum seed" > "$OUT"

echo "Ejecutando benchmark secuencial en head..."
echo "Salida: $OUT"
echo ""

for round in $(seq 1 "$ROUNDS"); do
  echo "Ronda secuencial $round/$ROUNDS..."

  for N in "${SIZES[@]}"; do
    echo "  N=$N"

    # Ejecutamos 1 trial por llamada.
    # El round externo controla las repeticiones.
    "$BIN" "$N" 1 "$SEED" | awk 'NR > 1' >> "$OUT"
  done
done

echo ""
echo "Benchmark secuencial completado."
echo "Resultados guardados en: $OUT"